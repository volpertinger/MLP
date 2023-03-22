import os
import settings as s
from dataset_utils import DataSet
from processing_utils import start_test_input

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = s.TF_CPP_MIN_LOG_LEVEL
    dataset = DataSet()
    dataset.train()
    start_test_input(dataset)
