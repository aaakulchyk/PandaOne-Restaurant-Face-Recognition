import argparse
import numpy as np
import pickle


from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import SVC


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="Path to serialized dataset of facial embeddings")
ap.add_argument("-r", "--recognizer", required=True,
                help="Path to output model trained to distinguish between faces")
ap.add_argument("-l", "--le", required=True,
                help="Path to output label encoder")
args = vars(ap.parse_args())


dataset = np.load(args["dataset"])
X, y = dataset["embeddings"], dataset["labels"]

# Normalize input vectors
enc_norm = Normalizer(norm='l2')
X = enc_norm.transform(X)

# Label encode targets
enc_label = LabelEncoder()
y = enc_label.fit_transform(y)

# Fit model
recognizer = SVC(kernel='linear', probability=True)
recognizer.fit(X, y)

# Write the actual face recognition SVM model to disk
with open(args["recognizer"], "wb") as f:
    f.write(pickle.dumps(recognizer))

# Write the label encoder to disk
with open(args["le"], "wb") as f:
    f.write(pickle.dumps(enc_label))
