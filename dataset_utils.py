import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow_datasets.core import visualization
from tensorflow_datasets.core.visualization import Visualizer

import settings as s
import matplotlib.pyplot as plt


class DataSet:

    def __init__(self):
        self.__ds, self.__ds_info = tfds.load(s.DATASET, split=['train', 'test'], with_info=True)
        self.__train = self.__ds[0]
        self.__test = self.__ds[1]
        self.__trained = True

    # ------------------------------------------------------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------------------------------------------------------
    def __show_image(self, index):
        visualization.image_visualizer.ImageGridVisualizer().show(self.__test.skip(index).take(1),
                                                                  self.__ds_info,
                                                                  rows=1,
                                                                  cols=1)

    # ------------------------------------------------------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------------------------------------------------------

    def get_info(self):
        return self.__ds_info

    def get_data_train_numpy(self):
        return tfds.as_numpy(self.__train)

    def get_data_test_numpy(self):
        return tfds.as_numpy(self.__test)

    def get_data_train_frame(self):
        return tfds.as_dataframe(self.__train, self.__ds_info)

    def get_data_test_frame(self):
        return tfds.as_dataframe(self.__test, self.__ds_info)

    def get_data_train_list(self):
        return list(self.__train)

    def get_data_test_list(self):
        return list(self.__test)

    def show_benchmark(self):
        tfds.benchmark(self.__ds, batch_size=s.BATCH_SIZE)

    def get_test_length(self):
        return len(self.__test)

    def show_examples(self):
        tfds.show_examples(self.__train, self.__ds_info)

    def predict(self, index):
        print(f"[predict] predicted value is {index}")
        self.__show_image(index)
