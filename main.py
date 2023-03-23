import os
import UserSettings as us
from MLP.MLP import MLP
from processing_utils import start_test_input

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = us.TF_CPP_MIN_LOG_LEVEL
    mlp = MLP(us.DATASET, us.MAX_EPOCHS, us.LEARNING_RATE, us.BATCH_SIZE, us.SAVE_FILENAME_MAIN, us.IMAGE_DIMENSION,
              us.WITH_INFO, us.USE_SPARE, us.SAVE_FILENAME_SPARE)
    mlp.train()
    start_test_input(mlp)
