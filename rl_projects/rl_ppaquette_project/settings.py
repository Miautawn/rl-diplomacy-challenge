import os

#####################
##
## DATA PROCESSING SETTINGS
## 
## These settings controll multiprocessing when extracting the data
## or creating the datasets. You should set them depending on your machine.
## These settings are provided, since by using concurrent processes,
## a bottleneck on data consumption (writing to dick) may appear,
## thus exploding the RAM!!!
##
## The default provided settings were used with a 4 core CPU and 15GB RAM.
#####################

# number of cpu workers to run concurrently when extracting the data
# None = use all!
DATA_EXTRACTION_NUM_WORKERS = None

# the buffer size for extracting the data
DATA_EXTRACTION_BUFFER_SIZE = 6000

# number of cpu workers to run concurrently when creating a dataset
# None = use all!
DATASET_CREATION_NUM_WORKERS = None

# the buffer size for creating a dataset
DATASET_CREATION_BUFFER_SIZE = 3000


#####################
##
## DATA PATH SETTINGS
##
## These settings simply tell where to store processable data / datasets.
#####################

# where are the extracted .jsonl files
EXTRACTED_DATA_DIR = "./data/raw"

# where the processed files should go
MODEL_DATA_DIR = "./data/model_data"

# all model-data related paths
MODEL_DATA_PATHS = {
    "UNPROCESSED_DATASET_PATH": os.path.join(MODEL_DATA_DIR, "unprocessed_dataset.txt"),
    "DATASET_INDEX_PATH" : os.path.join(MODEL_DATA_DIR, "dataset_index.pickle"),
    "END_SCS_DATASET_PATH" :  os.path.join(MODEL_DATA_DIR, "end_scs.pickle"),
    "HASH_DATASET_PATH" :  os.path.join(MODEL_DATA_DIR, "hash.pickle"),
    "MOVES_COUNT_DATASET_PATH" :  os.path.join(MODEL_DATA_DIR, "moves_count.pickle"),
    "PHASES_COUNT_DATASET_PATH" :  os.path.join(MODEL_DATA_DIR, "phases_count.pickle"),
    
    "TRAINING_DATASET_PATH": os.path.join(MODEL_DATA_DIR, "full_dataset_training.txt"),
    "VALIDATION_DATASET_PATH": os.path.join(MODEL_DATA_DIR, "full_dataset_validation.txt"),
    "TRAIN_VAL_DATA_COUNTS": os.path.join(MODEL_DATA_DIR, "full_dataset_train_val_counts.pickle")
}


#####################
##
## DATA FEATURE SETTINGS
##
## Settings for processing the data and creating the datasets.
## These are borrowed from the original dataset.
#####################
DATA_FEATURES = {
    "N_LOCATIONS": 81,
    "N_SUPPLY_CENTERS": 34,
    "N_LOCATION_FEATURES": 35,
    "N_ORDERS_FEATURES": 40,
    "N_POWERS": 7,
    "N_SEASONS": 3,
    "N_UNIT_TYPES": 2,

    "N_NODES": 81,                       # designed to be equal to N_LOCATIONS
    "TOKENS_PER_ORDER": 5,
    "MAX_LENGTH_ORDER_PREV_PHASES": 350,
    "MAX_CANDIDATES": 240,
    "N_PREV_ORDERS": 1,                  # We only feed the last movement phase
    "N_PREV_ORDERS_HISTORY": 3,          # We need to have an history of at least 3, to get at least 1 movement phase
}

VALIDATION_SET_SPLIT = 0.05


# dictionary of your features and their 'default' states.
#
# "static" means the feature has fixed length dimensions:
# - if the shape is empty list: [], then the feature is a single scalar
# - if the shape is specified, pad it to those dimensions
#
# "variable" means the feature has non-fixed length dimensions:
# - the shape doesn't matter in this case but you can mark it as [None].
# 
# this will be used to:
# - create missing features
# - pad static features to their max dimensions
# - pad variable dimension features within batches
DATA_BLUEPRINT = {
            'request_id': {"shape": [], "type": "static"},
            'player_seed':  {"shape": [], "type": "static"},
            'board_state':  {"shape": [81, 35], "type": "static"},
            'board_alignments': {"shape": [None], "type": "variable"},
            'prev_orders_state':  {"shape": [1, 81, 40], "type": "static"},
            'decoder_inputs':  {"shape": [None], "type": "variable"},
            'decoder_lengths':  {"shape": [], "type": "static"},
            'candidates':  {"shape": [None], "type": "variable"},
            'noise':  {"shape": [], "type": "static"},
            'temperature':  {"shape": [], "type": "static"},
            'dropout_rate':  {"shape": [], "type": "static"},
            'current_power': {"shape": [], "type": "static"},
            'current_season':  {"shape": [], "type": "static"},
            'draw_target':  {"shape": [], "type": "static"},
            'value_target':  {"shape": [], "type": "static"}
}


#####################
##
## TRAINING HYPERPARAMETER SETTINGS
##
## training/model related hyperparameters.
#####################
VALIDATION_SET_SPLIT = 0.05








