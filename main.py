import random

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from fgsm_attack import FGSM_Attack

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
test_images = test_images / 255.0
test_labels = to_categorical(test_labels)

model = tf.keras.models.load_model('mnist_model.keras')

epsilons = [0.1, 0.15, 0.2, 0.25, 0.3]

test_range = 100
test_index = random.randrange(len(test_images - test_range))

fgsm_attack = FGSM_Attack(model, test_images[test_index:test_index + test_range],
                          test_labels[test_index:test_index + test_range], epsilons)
fgsm_attack.run()

fgsm_attack.visualize()