from detection import extract_face

import argparse
import mtcnn
import numpy as np
import os
import pickle
import tensorflow as tf

from PIL import Image
from tensorflow import keras
from keras import models


ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', required=True,
                help='Path to FaceNet model that will extract face embeddings')
ap.add_argument('-d', '--dataset', required=True,
                help='Dataset that will be processed to extract face embeddings')
ap.add_argument('-s', '--save', required=False,
                help='Path to save compressed dataset')
args = vars(ap.parse_args())


def load_faces(folder):
    """Load images and extract faces for all images in `folder`"""

    faces = list()
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)

        # Get face
        face = extract_face(path)
        faces.append(face)
    return faces


def load_dataset(ds_folder):
    X, y = list(), list()

    for folder in os.listdir(ds_folder):
        path = os.path.join(ds_folder, folder)

        # Skip any files that might be in the directory
        if not os.path.isdir(path):
            continue

        # Load all faces in the subdirectory
        faces = load_faces(path)
        labels = [folder for _ in range(len(faces))]
        print(f'>loaded {len(labels)}/{len(faces)} examples for class: {folder}')

        X.extend(faces)
        y.extend(labels)
    return np.asarray(X), np.asarray(y)


def get_embedding(model, face_pixels):
    """Get the face embedding for one face"""

    # Scale pixel values
    face_pixels = face_pixels.astype(np.float32)

    # Standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std

    # Transform face into one sample
    samples = np.expand_dims(face_pixels, axis=0)

    # Make prediction to get embedding
    return model.predict(samples)[0]


if args['dataset'].endswith('.npz'):
    data = np.load(args['dataset'])
    X, y = data['embeddings'], data['labels']
else:
    X, y = load_dataset(args['dataset'])

model = models.load_model(args['model'])

X_prepared = list()
for face_pixels in X:
    embedding = get_embedding(model, face_pixels)
    X_prepared.append(embedding)
X_prepared = np.asarray(X_prepared)

save_path = args['save']
if not save_path.endswith('.npz'):
    save_path += '.npz'
np.savez_compressed(save_path, embeddings=X_prepared, labels=y)
