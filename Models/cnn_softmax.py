import tensorflow as tf
from tensorflow import keras

def build_cnn_softmax_model(input_shape, num_classes):

    model = tf.keras.Sequential()
    layers = tf.keras.layers

     #conv layer 1
    model.add(layers.Conv2D(
        filters=32,
        kernel_size=5,
        strides=1,
        activation=tf.nn.relu,
        kernel_initializer="he_normal"))
    model.add(layers.MaxPooling2D(strides=1, pool_size=2))
    #conv layer 1
    model.add(layers.Conv2D(
        filters=64,
        kernel_size=5,
        strides=1,
        activation=tf.nn.relu,
        kernel_initializer="he_normal",))
    model.add(layers.MaxPooling2D(strides=1, pool_size=2))

    # Flatten layers
    model.add(layers.Flatten())

    # FC layer 1
    model.add(layers.Dense(1024, activation=tf.nn.relu, kernel_initializer="he_normal"))

    #dropout layer
    model.add(layers.Dropout(rate=5e-1))  

    # FC 2 or Output layer
    model.add(layers.Dense(num_classes))
    return model

def compiled_cnn_softmax(model,alpha):
    model.compile(optimizer= tf.optimizers.Adam(learning_rate=alpha), loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    return model