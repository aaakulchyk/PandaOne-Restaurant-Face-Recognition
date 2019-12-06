from detection import extract_face

import argparse
import mtcnn
import numpy as np
import pickle
import os
import imutils


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Path to input image")
ap.add_argument("-d", "--detector", required=True,
                help="Path to MTCNN face detector")
ap.add_argument("-m", "--model", required=True,
                help="Path to Keras FaceNet embedding model")
ap.add_argument("-r", "--recognizer", required=True,
                help="Path to linear SVM recognizer")
ap.add_argument("-l", "--le", required=True,
                help="Path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="Minimum probability to filter weak detections")
args = vars(ap.parse_args())


detector = mtcnn.MTCNN()
