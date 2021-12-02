import tensorflow as tf
from keras.layers.merge import concatenate
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from input_data import load_data_label

num_filter = 4
num_class = 6901
batch_size = 128


def cnn(input):
    conv1_1 = layers.Conv2D(filters=num_filter, kernel_size=(5, 1))(input)
    conv1_2 = layers.Conv2D(filters=num_filter, kernel_size=(4, 1))(input)
    conv1_3 = layers.Conv2D(filters=num_filter, kernel_size=(3, 1))(input)
    conv1_4 = layers.Conv2D(filters=num_filter, kernel_size=(2, 1))(input)

    maxpool1_1 = layers.MaxPool2D((1, 1))(conv1_1)
    maxpool1_2 = layers.MaxPool2D((2, 1))(conv1_2)
    maxpool1_3 = layers.MaxPool2D((3, 1))(conv1_3)
    maxpool1_4 = layers.MaxPool2D((4, 1))(conv1_4)

    concated = concatenate([maxpool1_1, maxpool1_2, maxpool1_3, maxpool1_4], 1)

    fc1 = layers.Dense(1, name='Fc1')(concated)
    # fc1 = layers.MaxPool2D((4, 1))(concated)
    conv2 = layers.Conv2D(filters=num_filter, kernel_size=(2, 1))(fc1)

    maxpool2 = layers.MaxPool2D((3, 1))(conv2)

    fc2 = layers.Dense(1, name='Fc2')(maxpool2)

    return tf.reshape(fc2, (-1, num_class))


pos_weight = 0.0001


def top_k(y_true, y_pred):
    total_k = 0
    length = len(y_pred)
    for i in range(0, length):
        yp = y_pred[i]
        yt = y_true[i]
        yp, yt = zip(*sorted(zip(yp, yt)))
        if i == length - 1:
            print(yp)
            print(yt)
        # try:
        #     if yt[-20:].index(1.0) is not None:
        #         total_k += 1
        # except:
        #     pass
        if yt[-1] == 1.0:
            total_k += 1
    return total_k * 100.0 / length


def custom_loss():
    def loss(y_true, y_pred):
        weight_loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, pos_weight=pos_weight, labels=y_true)
        return tf.reduce_mean(weight_loss)

    return loss


if __name__ == '__main__':
    input_shape = (5, num_class, 1)
    input = Input(input_shape)
    output = cnn(input)
    model = Model(inputs=input, outputs=output)
    model.summary()

    # X_train, Y_train, X_val, Y_val, X_test, Y_test = load_data_label()
    #
    # adam = tf.keras.optimizers.Adam(learning_rate=0.001)
    # model.compile(optimizer=adam, loss=custom_loss())
    #
    # model_ = model.fit(tf.cast(X_train, tf.float32), tf.cast(Y_train, tf.float32),
    #                    validation_data=(tf.cast(X_val, tf.float32), tf.cast(Y_val, tf.float32)),
    #                    epochs=250, batch_size=batch_size)
    #
    # plt.plot(model_.history['val_loss'])
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.show()
    #
    # y_true = Y_train
    # y_pred = model.predict(tf.cast(X_train, tf.float32))
    # print(top_k(y_true, y_pred))
