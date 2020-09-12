import numpy as np
import struct
import random


class DatasetReader:
    def __init__(self, training_set, training_labels, testing_set, testing_labels):
        self.training_set = training_set
        self.training_labels = training_labels
        self.testing_set = testing_set
        self.testing_labels = testing_labels

    def get_training_dataset(self):
        with open(self.training_labels, 'rb') as labels_file:
            magic, size = struct.unpack(">II", labels_file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))

            self.train_labels = np.frombuffer(labels_file.read(), dtype=np.uint8)

        with open(self.training_set, 'rb') as image_file:
            magic, self.size, self.rows, self.columns = struct.unpack(">IIII", image_file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))

            self.train_images = np.zeros((self.size, self.rows, self.columns), dtype=np.uint8)
            for n in range(self.size):
                for j in range(self.columns):
                    self.train_images[n][j] = np.frombuffer(image_file.read(28), dtype=np.uint8)

        return self.train_images, self.train_labels

    def get_testing_dataset(self):
        with open(self.testing_labels, 'rb') as labels_file:
            magic, size = struct.unpack(">II", labels_file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))

            self.test_labels = np.frombuffer(labels_file.read(), dtype=np.uint8)

        with open(self.testing_set, 'rb') as image_file:
            magic, self.size, self.rows, self.columns = struct.unpack(">IIII", image_file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))

            self.test_images = np.zeros((self.size, self.rows, self.columns), dtype=np.uint8)
            for n in range(self.size):
                for j in range(self.columns):
                    self.test_images[n][j] = np.frombuffer(image_file.read(28), dtype=np.uint8)

        return self.test_images, self.test_labels

    def display(self, image=None):
        index = random.randint(0, self.size - 1)
        sample = self.train_images[index] if image is None else image
        for j in range(self.columns):
            for i in range(self.rows):
                print('.' if sample[j][i] <= 30 else '@', end='')
            print()

    def get_shape(self):
        return self.rows, self.columns, 1

    def get_number_of_classes(self):
        return 10
