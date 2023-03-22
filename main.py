from tensorflow_datasets.core import dataset_info

from dataset_utils import DataSet
import tensorflow_datasets as tfds
import matplotlib

if __name__ == '__main__':
    dataset = DataSet()
    # dataset.show_benchmark()
    dataset.show_examples()
    print("end")
