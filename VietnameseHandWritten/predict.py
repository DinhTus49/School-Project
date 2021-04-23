from tensorflow.keras.models import load_model
import argparse
import os
import preprocess
import numpy as np

SAVED_MODEL_FOLDER = "SavedModel"

parser = argparse.ArgumentParser(description='predict the image input')
parser.add_argument('--image_path', type=str, help='the path to the image')
parser.add_argument('--model_name', type=str, help='the name of the model')
args = parser.parse_args()

model_path = os.path.join(SAVED_MODEL_FOLDER, args.model_name)

model = load_model(model_path)
pred = model.predict(args.image_path)
classIndex = np.argmax(pred, axis=1)
labels = preprocess.LABELS
label = labels[classIndex]

print("predict: %s" % label)