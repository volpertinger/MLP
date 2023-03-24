# ----------------------------------------------------------------------------------------------------------------------
# Hyper parameters
# ----------------------------------------------------------------------------------------------------------------------
BATCH_SIZE = 64
MAX_EPOCHS = 30
LEARNING_RATE = 0.00024
IMAGE_DIMENSION = 3

# ----------------------------------------------------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------------------------------------------------
DATASET = "horses_or_humans"
__DIRECTORY_PREFIX = "saves"
SAVE_FILENAME_MAIN = f"{__DIRECTORY_PREFIX}_main/{DATASET}"
USE_SPARE = True
SAVE_FILENAME_SPARE = f"{__DIRECTORY_PREFIX}_spare/{DATASET}"
# ----------------------------------------------------------------------------------------------------------------------
# System
# ----------------------------------------------------------------------------------------------------------------------
TF_CPP_MIN_LOG_LEVEL = '3'

# ----------------------------------------------------------------------------------------------------------------------
# Other
# ----------------------------------------------------------------------------------------------------------------------
WITH_INFO = True
