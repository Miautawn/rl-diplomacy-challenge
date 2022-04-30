import logging
import pickle
from collections import OrderedDict
import concurrent.futures
from queue import Queue

from tqdm import tqdm


from utilities.dataset_creation_functions import (get_request_id, 
                                                 get_policy_data,
                                                 format_datapoint,
                                                 generate_datapoint)

from utilities.data_extraction_functions import decompress_game
from utilities.utility_functions import compress_dict, decompress_dict

from settings import (DATASET_CREATION_NUM_WORKERS, 
                      DATASET_CREATION_BUFFER_SIZE,
                      MODEL_DATA_PATHS,
                      MODEL_DATA_DIR,
                      DATA_FEATURES,
                      H_PARAMS)

LOGGER = logging.getLogger(__name__)


def process_sample(compressed_game_string):
    """ Handles the datapoint processing"""
    try:
        saved_game = decompress_game(compressed_game_string)
        is_validation_sample = bool(saved_game["id"] in VALID_IDS)
        
        game_results = generate_datapoint(saved_game, is_validation_sample)
        results = []
        for phase_idx in range(len(saved_game["phases"])):
            if not game_results[phase_idx]:
                results.append((is_validation_sample, None, None))
                continue
            for power_name in game_results[phase_idx]:
                _, game_result = game_results[phase_idx][power_name]
                results.append((is_validation_sample, power_name, game_result))
        return (saved_game["id"], results)
    except Exception as exc:
        raise exc
        

# train/val datapoint counts
dataset_index = {
    'size_train_dataset' : 0,
    'size_valid_dataset': 0
}

# Splitting games into training and validation
phase_count_dataset = pickle.load(open(MODEL_DATA_PATHS["PHASES_COUNT_DATASET_PATH"], 'rb'))      
game_ids = list(phase_count_dataset.keys())

n_valid = int(VALIDATION_SET_SPLIT * len(game_ids))
set_valid_ids = set(sorted(game_ids)[-n_valid:])

TRAIN_IDS = list([game_id for game_id in game_ids if game_id not in set_valid_ids])
VALID_IDS = list(set_valid_ids)

# opening data & training/validation files for reading/writing
all_samples = open(MODEL_DATA_PATHS["UNPROCESSED_DATASET_PATH"], "r")
train_dataset = open(MODEL_DATA_PATHS["TRAINING_DATASET_PATH"], "w")
valid_dataset = open(MODEL_DATA_PATHS["VALIDATION_DATASET_PATH"], "w")


current_samples = 0
pending_samples = len(game_ids)
pending_phases = sum(phase_count_dataset.get(game_id, 0) for game_id in game_ids)

progress_bar = tqdm(total=pending_phases)

# main loop
with concurrent.futures.ProcessPoolExecutor(max_workers=DATASET_CREATION_NUM_WORKERS) as executor:
    while pending_samples != 0:
        
        # reading in compressed samples to the buffer
        local_buffer = []
        local_buffer_size = min(pending_samples, DATASET_CREATION_BUFFER_SIZE)
        for i in range(local_buffer_size):
            local_buffer.append(all_samples.readline())
            
        # process the datapoints stored in the buffer
        for processed_sample in executor.map(process_sample, local_buffer):
            
            game_id, processed_datapoints = processed_sample
            # each game sample produces results for each phase and power
            for is_validation_datapoint, power_name, game_result in processed_datapoints:
                if game_result is not None:
                    
                    if(is_validation_datapoint):
                        valid_dataset.write(compress_dict(game_result) + "\n")
                        dataset_index['size_valid_dataset'] += 1
                    else:
                        train_dataset.write(compress_dict(game_result) + "\n")
                        dataset_index['size_train_dataset'] += 1
                    
            progress_bar.update(phase_count_dataset.get(game_id))
            pending_samples -= 1
            
        # free the memory of the buffer
        del local_buffer
    
# writing train/val datapoint counts 
pickle.dump(dataset_index, open(MODEL_DATA_PATHS["TRAIN_VAL_DATA_COUNTS"], "wb"))           

progress_bar.close()    
all_samples.close()
train_dataset.close()
valid_dataset.close()