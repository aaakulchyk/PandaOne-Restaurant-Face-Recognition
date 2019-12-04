import cv2 as cv
import imutils
import json
import numpy as np
import os
import pickle
import urllib

from collections import Iterable

from django.http import JsonResponse
from django.shortcuts import render
from django.views import generic
from django.views.decorators.csrf import csrf_exempt

from . import models as _models


FACE_DETECTOR_PATH = "{base_path}/cascades/haarcascade_frontalface_default.xml".format(
    base_path=os.path.abspath(os.path.dirname(__file__)))

RECOGNITION_PARAMS = {
        "recognizer": "face_detection/output/recognizer.pickle",
        "detector": "face_detection/face_detection_model",
        "embedding_model": "face_detection/openface_nn4.small2.v1.t7",
        "le": "face_detection/output/recognizer.pickle",
        "confidence": 0.8
    }

RECOGNITION_CACHE = set()


def index(request, *args, **kwargs):
    return render(request, 'face_detection/index.html')


def face_recognition(request, *args, **kwargs):
    return render(request, 'face_detection/face_recognition.html')


def json_customer(request, pk, *args, **kwargs):
    return JsonResponse({'customer': _query_recognized_customer(pk-1)})


def test(request, *args, **kwargs):
    return render(request, 'face_detection/test.html')


class CustomerListView(generic.ListView):
    model = _models.Customer


class CustomerDetailView(generic.DetailView):
    model = _models.Customer
    template_name = 'face_detection/customer_detail.html'
    context_object_name = 'customer'


class RegularCustomerListView(generic.ListView):
    model = _models.RegularCustomer


class RegularCustomerDetailView(generic.DetailView):
    model = _models.RegularCustomer
    template_name = 'face_detection/regular_customer_detail.html'
    context_object_name = 'customer'


@csrf_exempt
def detect(request, *args, **kwargs):
    # initialize the data dictionary to be returned by the request
    data = {"success": False}

    # check to see if this is a post request
    # print(request.method)
    if request.method == "POST":
        # check to see if an image was uploaded
        if request.FILES.get("image", None) is not None:
            # grab the uploaded image
            image = _grab_image(stream=request.FILES["image"])
            # print(f"Image: {image}")

        # otherwise, assume that a URL was passed in
        else:
            # grab the URL from the request
            url = request.POST.get("url", None)

            # if the URL is None, then return an error
            if url is None:
                data["error"] = "No URL provided."
                return JsonResponse(data)

            # load the image and convert
            image = _grab_image(url=url)

        # convert the image to grayscale, load the face cascade detector,
        # and detect faces in the image
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        detector = cv.CascadeClassifier(FACE_DETECTOR_PATH)
        # print(image)
        rects = detector.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5,
                                          minSize=(30, 30), flags=cv.CASCADE_SCALE_IMAGE)

        # construct a list of bounding boxes from the detection
        rects = [(int(x), int(y), int(x + w), int(y + h)) for (x, y, w, h) in rects]

        # update the data dictionary with the faces detected
        data.update({"num_faces": len(rects), "faces": rects, "success": True})

    # return a JSON response
    return JsonResponse(data)


@csrf_exempt
def detect_async(request, *args, **kwargs):
    # initialize the data dictionary to be returned by the request
    data = {"success": False}

    # check to see if this is a post request
    if request.method == "POST":
        # check to see if an image was uploaded
        # print([i for i in dir(request.POST) if not i.startswith('_')])
        # print("POST", request.POST)
        # print("FILES", request.FILES)
        if request.FILES.get("image", None) is not None:
            # grab the uploaded image
            image = _grab_image(stream=request.FILES.get("image"))
            # print(f"Image: {image}")

        # otherwise, assume that a URL was passed in
        else:
            # grab the URL from the request
            url = request.POST.get("url", None)

            # if the URL is None, then return an error
            if url is None:
                data["error"] = "No URL provided."
                return JsonResponse(data)

            # load the image and convert
            image = _grab_image(url=url)

        detections, names = _recognize(image, **RECOGNITION_PARAMS)
        # print(f"Detections: {detections}")
        customers = _query_recognized_customers(names)

        # Cache customers that were recognized at the picture
        _cache_recognition_results(names)

        data.update({"success": True, "num_faces": detections.shape[0], "customers": customers})
        # print(f"DATA: {data}")

    # return a JSON response
    return JsonResponse(data)


def _cache_recognition_results(names):
    for name in names:
        RECOGNITION_CACHE.add(name)


def _grab_image(path=None, stream=None, url=None):
    # if the path is not None, then load the image from disk
    if path is not None:
        image = cv.imread(path)

    # otherwise, the image does not reside on disk
    else:
        # if the URL is not None, then download the image
        if url is not None:
            resp = urllib.request.urlopen(url)
            data = resp.read()

        # if the stream is not None, then the image has been uploaded
        elif stream is not None:
            data = stream.read()

        # convert the image to a NumPy array and then read it into
        # OpenCV format
        image = np.asarray(bytearray(data), dtype="uint8")
        image = cv.imdecode(image, cv.IMREAD_COLOR)

    # return the image
    return image


def _recognize(image, **kwargs):
    # load our serialized face detector from disk
    # print("[INFO] loading face detector...")
    proto_path = os.path.sep.join([kwargs["detector"], "deploy.prototxt"])
    model_path = os.path.sep.join([kwargs["detector"],
                                  "res10_300x300_ssd_iter_140000.caffemodel"])

    assert os.path.exists(proto_path), f"Proto path does not exist: {proto_path}"
    assert os.path.exists(model_path), f"Model path does not exist: {model_path}"

    detector = cv.dnn.readNetFromCaffe(proto_path, model_path)

    # load our serialized face embedding model from disk
    # print("[INFO] loading face recognizer...")
    embedder = cv.dnn.readNetFromTorch(kwargs["embedding_model"])

    # load the actual face recognition model along with the label encoder
    recognizer = pickle.loads(open(kwargs["recognizer"], "rb").read())
    le = pickle.loads(open(kwargs["le"], "rb").read())

    # load the image, resize it to have a width of 600 pixels (while
    # maintaining the aspect ratio), and then grab the image dimensions
    # image = cv.imread(kwargs["image"])
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    # construct a blob from the image
    image_blob = cv.dnn.blobFromImage(
        cv.resize(image, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(image_blob)
    detections = detector.forward()
    # print(f"Detections shape: {detections.shape}")
    # print(detections[0, 0, :, 2].shape)
    # print(kwargs["confidence"])

    names = list()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > kwargs["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue

            # construct a blob for the face ROI, then pass the blob
            # through our face embedding model to obtain the 128-d
            # quantification of the face
            faceBlob = cv.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
                                            (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # perform classification to recognize the face
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]
            # print(f"Classes: {le.classes_}")
            names.append(str(name))

            # draw the bounding box of the face along with the associated
            # probability
            text = "{}: {:.2f}%".format(name, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            """cv.rectangle(image, (startX, startY), (endX, endY),
                         (0, 0, 255), 2)
            cv.putText(image, text, (startX, y),
                       cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)"""

    return detections[0, 0, detections[0, 0, :, 2] > kwargs['confidence']], names


def _query_recognized_customers(names, fields='all'):
    return [_query_recognized_customer(name) for name in names]


def _query_recognized_customer(name, fields='all'):
    if fields != 'all' and not isinstance(fields, Iterable):
        raise ValueError(f"Argument `fields` must be equal 'all' or be iterable")

    customer = _models.Customer.objects.get(pk=int(name)+1)
    if fields == 'all':
        return {
            'id': customer.pk,
            'name': customer.name,
            'sex': customer.sex,
            'first_visit': customer.first_visit,
            'spent': customer.spent,
        }
