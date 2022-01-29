import os

N_LOCATIONS = 81
N_SUPPLY_CENTERS = 34
N_LOCATION_FEATURES = 35
N_ORDERS_FEATURES = 40
N_POWERS = 7
N_SEASONS = 3
N_UNIT_TYPES = 2

N_NODES = N_LOCATIONS
TOKENS_PER_ORDER = 5
MAX_LENGTH_ORDER_PREV_PHASES = 350
MAX_CANDIDATES = 240
N_PREV_ORDERS = 1                  # We only feed the last movement phase
N_PREV_ORDERS_HISTORY = 3          # We need to have an history of at least 3, to get at least 1 movement phase

VALIDATION_SET_SPLIT = 0.05


# dictionary of your features and their 'default' states
# this will be used to fill empty or create missing features
DATA_BLUEPRINT = {
            'request_id': {"shape": [], "type": "static"},
            'player_seed':  {"shape": [], "type": "static"},
            'board_state':  {"shape": [81, 35], "type": "static"},
            'board_alignments': {"shape": [2754], "type": "variable"},
            'prev_orders_state':  {"shape": [1, 81, 40], "type": "static"},
            'decoder_inputs':  {"shape": [35], "type": "variable"},
            'decoder_lengths':  {"shape": [], "type": "static"},
            'candidates':  {"shape": [None, 240], "type": "variable"},
            'noise':  {"shape": [], "type": "static"},
            'temperature':  {"shape": [], "type": "static"},
            'dropout_rate':  {"shape": [], "type": "static"},
            'current_power': {"shape": [], "type": "static"},
            'current_season':  {"shape": [], "type": "static"},
            'draw_target':  {"shape": [], "type": "static"},
            'value_target':  {"shape": [], "type": "static"}
}


EXTRACTED_DATA_DIR = "./data/raw"
MODEL_DATA_DIR = "./data/model_data"

MODEL_DATA_PATHS = {
    "MAIN_DATASET_PATH": os.path.join(MODEL_DATA_DIR, "full_dataset.txt"),
    "DATASET_INDEX_PATH" : os.path.join(MODEL_DATA_DIR, "dataset_index.pickle"),
    "END_SCS_DATASET_PATH" :  os.path.join(MODEL_DATA_DIR, "end_scs.pickle"),
    "HASH_DATASET_PATH" :  os.path.join(MODEL_DATA_DIR, "hash.pickle"),
    "MOVES_COUNT_DATASET_PATH" :  os.path.join(MODEL_DATA_DIR, "moves_count.pickle"),
    "PHASES_COUNT_DATASET_PATH" :  os.path.join(MODEL_DATA_DIR, "phases_count.pickle"),
    
    "TRAINING_DATASET_PATH": os.path.join(MODEL_DATA_DIR, "full_dataset_training.txt"),
    "VALIDATION_DATASET_PATH": os.path.join(MODEL_DATA_DIR, "full_dataset_validation.txt"),
    "INDEX_DATASET_PATH": os.path.join(MODEL_DATA_DIR, "full_dataset_index.pickle")
}




