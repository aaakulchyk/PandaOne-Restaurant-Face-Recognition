import requests
import cv2 as cv


# define the URL to our face detection API
url = "http://localhost:8000/face_recognition/"

# use our face detection API to find faces in images via image URL
image = cv.imread("images/obama.jpg")
payload = {"url": "https://www.pyimagesearch.com/wp-content/uploads/2015/05/obama.jpg"}
r = requests.post(url, data=payload).json()
print("obama.jpg: {}".format(r))

# loop over the faces and draw them on the image
for (startX, startY, endX, endY) in r["faces"]:
    cv.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

# show the output image
cv.imshow("obama.jpg", image)
cv.waitKey(0)

# load our image and now use the face detection API to find faces in
# images by uploading an image directly
image = cv.imread("images/adrian.jpg")
payload = {"image": open("images/adrian.jpg", "rb")}
r = requests.post(url, files=payload).json()
print("adrian.jpg: {}".format(r))

# loop over the faces and draw them on the image
for (startX, startY, endX, endY) in r["faces"]:
    cv.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

# show the output image
cv.imshow("adrian.jpg", image)
cv.waitKey(0)
