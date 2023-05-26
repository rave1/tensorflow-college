import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from random import randint


asl_alphabets = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
    5: 'F', 6: 'G', 7: 'H', 8: 'I',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
    15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y'
}

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
    # while True:
    #     try:
    #         rand = randint(0,len(x_test-1))
    #         prediction = model.predict(x_test[rand][np.newaxis, ..., np.newaxis])
    #         print(prediction)
    #         print(np.argmax(prediction))
    #         print(f"The sign is probably a {asl_alphabets[np.argmax(prediction)]}")
    #         fig = plt.figure(figsize=(10,7))
    #         plt.axis(False)
    #         plt.title(f"I see {asl_alphabets[np.argmax(prediction)]}")
    #         plt.imshow(x_test[rand])
    #         plt.show()
    #     except Exception as e:
    #         print(f"Error reading image! Proceeding with next image, error: {e}")
    #         break


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
