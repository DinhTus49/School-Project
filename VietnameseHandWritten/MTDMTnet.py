import tensorflow as tf

img_size = (32, 32)


class MTDMTnet:
    @staticmethod
    def build(height, width, classes):
        global img_size
        if img_size != (32, 32):
            img_size = (height, width)
        model = tf.keras.models.Sequential()
        input_shape = (height, width, 1)

        model.add(tf.keras.layers.Conv2D(6, (5, 5), input_shape=input_shape))
        model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))

        model.add(tf.keras.layers.Conv2D(16, (5, 5)))
        model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))

        model.add(tf.keras.layers.Flatten())

        model.add(tf.keras.layers.Dense(120))
        model.add(tf.keras.layers.Activation(tf.keras.activations.relu))

        model.add(tf.keras.layers.Dense(84))
        model.add(tf.keras.layers.Activation(tf.keras.activations.relu))

        model.add(tf.keras.layers.Dense(classes))
        model.add(tf.keras.layers.Activation(tf.keras.activations.softmax))

        return model
