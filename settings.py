# ----------------------------------------------------------------------------------------------------------------------
# Hyper parameters
# ----------------------------------------------------------------------------------------------------------------------
BATCH_SIZE = 32
LAYERS_NUMBER = 3
NEURON_NUMBER = 4
ACTIVATION = "elu"
LOSS_FUNCTION = "adam"
MAX_EPOCHS = 10
LEARNING_RATE = 0.00024

# ----------------------------------------------------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------------------------------------------------
DATASET = "horses_or_humans"
SAVE_FILENAME_MAIN = f"model_saves/{DATASET}"
USE_SPARE = True
SAVE_FILENAME_SPARE = f"model_saves_spare/{DATASET}"

# ----------------------------------------------------------------------------------------------------------------------
# System
# ----------------------------------------------------------------------------------------------------------------------
TF_CPP_MIN_LOG_LEVEL = '3'

# ----------------------------------------------------------------------------------------------------------------------
# Other
# ----------------------------------------------------------------------------------------------------------------------
WITH_INFO = False
