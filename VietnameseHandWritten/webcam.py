from model_misc import predict
import cv2
import numpy as np
import argparse
import preprocess
# label for predict
labels = preprocess.LABELS

parser = argparse.ArgumentParser(description='predict the vietnamese handwritten by webcam')
parser.add_argument('--model_name', type=str, help='the name of model')
args = parser.parse_args()

# name of the model
model_name = args.model_name

# width height of the video
framewidth = 640
frameheight = 480

cap = cv2.VideoCapture(0)
cap.set(3, framewidth)
cap.set(4, frameheight)
# the bright of the video
cap.set(10, 150)

# convert the image to gray scale
def preProcess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # balance image brightness
    img = cv2.equalizeHist(img)
    img = img/255
    return img


while True:
    ret, frame_original = cap.read()
    frame = np.asarray(frame_original)
    frame = cv2.resize(frame, (32,32))
    frame = preProcess(frame)

    # predict the input
    frame = frame.reshape(32, 32, 1)
    pred = predict(frame, model_name)
    classIndex = np.argmax(pred, axis=1)
    label = labels[classIndex]
    print(label)

    cv2.putText(frame_original, str(label), (50, 50), cv2.FONT_ITALIC, 1, (0,255,0), 1)
    cv2.imshow("web cam", frame_original)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()