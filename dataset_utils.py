import tensorflow_datasets as tfds
import tensorflow as tf
import settings as s

ds = tfds.load(s.DATASET, split='train', as_supervised=True)
