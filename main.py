import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras_visualizer import visualizer
from keras.utils.vis_utils import plot_model
import visualkeras
import string



def test_model(model, x_test, y_test) -> None:
    LETTER_MAPPER = dict(zip(string.ascii_uppercase, range(0,26)))
    predictions = np.argmax(model.predict(x_test), axis=1)
    print(len(predictions))
    for i in range(len(predictions)):
        if(predictions[i] >= 9):
            predictions[i] += 1
    i = 0
    for c in predictions[:24]:
        plt.subplot(6, 4, i+1)
        plt.imshow(x_test[c], cmap='gray', interpolation='none')
        plt.title(f'Predicted letter: {list(filter(lambda x: LETTER_MAPPER[x] == predictions[c], LETTER_MAPPER))[0]}, Actual: {list(filter(lambda x: LETTER_MAPPER[x] == y_test[c], LETTER_MAPPER))[0]}')
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
    # visualizer(model=model, file_format='png', view=True)
    # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    # visualkeras.layered_view(model, to_file='output.png').show()


if __name__ == '__main__':
    main()
