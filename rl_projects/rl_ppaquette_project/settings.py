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
    "unprocessed_dataset_path": os.path.join(MODEL_DATA_DIR, "unprocessed_dataset.txt"),
    "dataset_index_path" : os.path.join(MODEL_DATA_DIR, "dataset_index.pickle"),
    "end_supply_center_dataset_path" :  os.path.join(MODEL_DATA_DIR, "end_scs.pickle"),
    "hash_dataset_path" :  os.path.join(MODEL_DATA_DIR, "hash.pickle"),
    "moves_count_dataset_path" :  os.path.join(MODEL_DATA_DIR, "moves_count.pickle"),
    "phases_count_dataset_path" :  os.path.join(MODEL_DATA_DIR, "phases_count.pickle"),
    
    "training_dataset_path": os.path.join(MODEL_DATA_DIR, "full_dataset_training.txt"),
    "validation_dataset_path": os.path.join(MODEL_DATA_DIR, "full_dataset_validation.txt"),
    "train_validation_data_counts": os.path.join(MODEL_DATA_DIR, "full_dataset_train_val_counts.pickle")
}


#####################
##
## DATA FEATURE SETTINGS
##
## Settings for processing the data and creating the datasets.
## These are borrowed from the original dataset.
#####################
DATA_FEATURES = {
    "n_locations": 81,
    "n_supply_centers": 34,
    "n_location_features": 35,
    "n_orders_features": 40,
    "n_powers": 7,
    "n_seasons": 3,
    "n_unit_types": 2,

    "n_nodes": 81,                       # designed to be equal to N_LOCATIONS
    "tokens_per_order": 5,
    "max_length_order_prev_phases": 350,
    "max_candidates": 240,
    "n_prev_orders": 1,                  # We only feed the last movement phase
    "n_prev_orders_history": 3,          # We need to have an history of at least 3, to get at least 1 movement phase
}

VALIDATION_SET_SPLIT = 0.05


# dictionary of your input features and their 'default' states.
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
H_PARAMETERS = {
    'batch_size': 128,
    'sync_gradients': True,
    'avg_gradients': False,
    'grad_aggregation': 'ADD_N',
    'use_partitioner': False,
    'use_verbs': False,
    'learning_rate': 0.001,
    'lr_decay_factor': 0.93,
    'max_gradient_norm': 5.0,
    'beam_width': 10,
    'beam_groups': 5,
    'dropout_rate': 0.5,
    'use_v_dropout': True,
    'perc_epoch_for_training': 1.0,
    'early_stopping_stop_after': 5,
    'policy_coeff': 1.0,
    'n_graph_conv': 16,
    'order_emb_size': 80,
    'power_emb_size': 60,
    'season_emb_size': 20,
    'gcn_size': 120,
    'lstm_size': 200,
    'attn_size': 120,
    'validation_set_split': 0.05,
}








