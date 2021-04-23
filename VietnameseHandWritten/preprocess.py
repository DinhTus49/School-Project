import cv2
import numpy as np
import os, sys

LABELS = ["a", "ă", "â", "b", "c", "d", "đ", "e", "ê", "g", "h",
          "i", "k", "l", "m", "n", "o", "ô", "ơ", "p", "q", "r",
          "s", "t", "u", "ư", "v", "x", "y"]
DEFAULT_DATA_DIR = "Data"
DEFAULT_RESULT_DIR = "Results"


def preprocess(labels=LABELS, data_dir=DEFAULT_DATA_DIR, result_dir=DEFAULT_RESULT_DIR):
    assert labels, "Please specify labels"
    for label in labels:
        counter = 0
        data_path = os.path.join(data_dir, label)
        result_path = os.path.join(result_dir, label)
        try:
            for file in os.listdir(data_path):
                image_path = os.path.join(data_path, file)
                stream = open(image_path, 'rb')
                bytes = bytearray(stream.read())
                np_array = np.asarray(bytes, dtype=np.uint8)
                image = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)
                try:
                    os.makedirs(result_path, exist_ok=True)
                except OSError as e:
                    print("Directory '%s' can not be created" %result_path)
                is_success, im_buf_arr = cv2.imencode(".jpg", image)
                im_buf_arr.tofile(os.path.join(result_path, str(label) + str(counter) + ".jpg"))
                #result_image = cv2.imwrite('r %s' %os.path.join(result_path, str(label) + str(counter) + ".jpg"), image)
                counter += 1
        except FileNotFoundError as e:
            pass


if __name__ == '__main__':
    preprocess()
