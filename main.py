import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


def test_model(model, x_test, y_test):
    predictions = np.argmax(model.predict(x_test), axis=1)
    for i in range(len(predictions)):
        if(predictions[i] >= 9):
            predictions[i] += 1
    correct = np.nonzero(predictions == y_test)[0]
    i = 0
    for c in correct[:6]:
        plt.subplot(3, 2, i+1)
        plt.imshow(x_test[c], cmap='gray', interpolation='none')
        plt.title(f'Predicted class: {predictions[c]}, Actual {y_test[c]}')
        plt.tight_layout()
        i += 1
    plt.show()


def load_data() -> tuple:
    test_data = pd.read_csv('./data/sign_mnist_test.csv', index_col=False)
    x_test_pd = test_data.drop('label', axis=1)
    y_test_pd = test_data['label']

    x_test = tf.constant(x_test_pd,shape=(7172,28,28), dtype=tf.float32)
    y_test = tf.constant(y_test_pd, dtype=tf.int32)
    x_test = x_test / 255
    return x_test, y_test


def main() -> None:
    model = tf.keras.models.load_model('new_sign_model.h5')
    model.summary()
    a, b = load_data()
    test_model(model, a, b)


if __name__ == '__main__':
    main()
