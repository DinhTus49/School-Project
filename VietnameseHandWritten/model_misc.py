from MTDMTnet import MTDMTnet, img_size
# import MTDMTnet
import cv2
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import os, sys
import datetime
import argparse
from tensorboard import program
import webbrowser


datas = []
labels = []
img_size = img_size
X_train = np.array([])
X_test = np.array([])
y_train = np.array([])
y_test = np.array([])

SAVED_MODEL_FOLDER = "SavedModel"
BATCH_SIZE = 8
EPOCHS = 50

le = LabelEncoder()


def load_in_data(data_dir):
    global img_size
    global datas
    global labels
    global le
    for label in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label)
        for file in os.listdir(label_path):
            image_path = os.path.join(label_path, file)
            try:
                stream = open(image_path, 'rb')
                bytes = bytearray(stream.read())
                np_array = np.asarray(bytes, dtype=np.uint8)
                image = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)
                assert image is not None
                image = cv2.resize(image, img_size)
                image = np.reshape(image, (img_size[0], img_size[1], 1))
                datas.append(image)
                labels.append(label)
            except AssertionError as error:
                error.args += ("%s is not an image file" % file,)
    datas = np.array(datas, dtype=np.float) / 255.0
    labels = le.fit_transform(labels)
    labels = tf.keras.utils.to_categorical(labels)


def train(validation_split: int, tensor_board_log_dir=None):
    global X_train, X_test, y_train, y_test
    global SAVED_MODEL_FOLDER
    global EPOCHS
    (X_train, X_test, y_train, y_test) = train_test_split(datas, labels, test_size=0.2)
    model = MTDMTnet.build(img_size[0], img_size[1], len(le.classes_))
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    model_timer = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if tensor_board_log_dir is not None and type(tensor_board_log_dir) == str:
        log_dir = os.path.join(tensor_board_log_dir, "MTDMTnet " + model_timer)
        tensor_board_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        history = model.fit(X_train, y_train, epochs=EPOCHS, verbose=2, validation_split=validation_split,
                            callbacks=tensor_board_callback)
    else:
        history = model.fit(X_train, y_train, epochs=EPOCHS, verbose=2, validation_split=validation_split)
    model.save(os.path.join(SAVED_MODEL_FOLDER, "MTDMTnet " + model_timer), save_format="h5")


# vẽ đồ thị loss/accuracy của train/validation set theo mỗi epoch
# thêm vào loss, accuracy trên test set
def evaluate(model_name, tensorBoard_path, batch_size=None):
    right_prediction = 0
    total_prediction = 0
    test_acc = 0.0
    try:
        assert model_name
        # for i in range(0, X_test.shape)
        model = tf.keras.models.load_model(os.path.join(SAVED_MODEL_FOLDER, model_name))
        predictions = model.predict(x=X_test, batch_size=batch_size)
        print(classification_report(y_test.argmax(axis=1),
                                    predictions.argmax(axis=1),
                                    target_names=le.classes_))
        print("Accuracy of test_set: %d" % (predictions.argmax(axis=1)/y_test.argmax(axis=1))*100)

    except AssertionError as err:
        err.args += ("Please specify model name",)

    # Run the tensor board
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tensorBoard_path])
    url = tb.launch()
    webbrowser.open(url)

def predict(img, model_name):
    try:
        assert model_name
        model = tf.keras.models.load_model(os.path.join(SAVED_MODEL_FOLDER, model_name))
        pred = model.predict(img)

        return pred
    except AssertionError as err:
        err.args += ("Please specify model name",)


def main():
    parser = argparse.ArgumentParser(description='load data from data directory')
    parser.add_argument('--data_dir', type=str, help='the direction contain data')
    parser.add_argument('--validation_split', type=float, help='rate for validation dataset')
    parser.add_argument('--tensorboard_path', type=str, help='path to tensor board')
    parser.add_argument('--_evaluate', type=int, default=0, help='want to evaluate or not?')
    args = parser.parse_args()

    load_in_data(args.data_dir)
    train(args.validation_split, args.tesorboard_path)


if __name__ == '__main__':
    main()
