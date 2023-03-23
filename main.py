import os
import settings as s
from dataset_utils import MLP
from processing_utils import start_test_input

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = s.TF_CPP_MIN_LOG_LEVEL
    mlp = MLP(s.DATASET, s.MAX_EPOCHS, s.LEARNING_RATE, s.BATCH_SIZE, s.SAVE_FILENAME_MAIN, s.WITH_INFO)
    mlp.train()
    start_test_input(mlp)
