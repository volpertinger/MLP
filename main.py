from dataset_utils import DataSet
from processing_utils import start_test_input

if __name__ == '__main__':
    dataset = DataSet()
    dataset.train_model()
    start_test_input(dataset)
