import os
from collections import OrderedDict
import concurrent.futures
import glob
import json
import logging
import pickle
from queue import Queue
import sys


from diplomacy import Game, Map
from tqdm import tqdm
import numpy as np

from utilities.reward_functions import DefaultRewardFunction
from utilities.data_extraction_functions import (convert_to_board_representation,
                                                convert_to_previous_orders_representation,
                                                add_cached_states_to_saved_game,
                                                add_possible_orders_to_saved_game,
                                                add_rewards_to_saved_game,
                                                get_end_scs_info, get_moves_info,
                                                compress_game, decompress_game)

from utilities.utility_functions import get_map_powers

from settings import (DATA_EXTRACTION_NUM_WORKERS, 
                      DATA_EXTRACTION_BUFFER_SIZE,
                      MODEL_DATA_DIR,
                      DATA_FEATURES,
                      H_PARAMS)


LOGGER = logging.getLogger(__name__)

def process_game(line):
    """ Process a line in the .jsonl file
        :return: A tuple (game_id, saved_game_zlib)
    """
    if not line:
        return None, None
    saved_game = json.loads(line)
    saved_game = add_cached_states_to_saved_game(saved_game)
    saved_game = add_possible_orders_to_saved_game(saved_game)
    saved_game = add_rewards_to_saved_game(saved_game, DefaultRewardFunction())
    return saved_game["id"], saved_game


def main():
    
    #making a dir for extracted data if it does not exits
    if not os.path.isdir(MODEL_DATA_DIR):
        os.makedirs(MODEL_DATA_DIR)
        print(f"Created a dir for data extraction: {MODEL_DATA_DIR}")
        LOGGER.info(f"Created a dir for data extraction: {MODEL_DATA_DIR}")
    
    # defining some parameters which are required for data construction
    map_object = Map()
    all_powers = get_map_powers(map_object)
    supply_centers_to_win = len(map_object.scs) // 2 + 1

    # defining iterables that will hold information which will be extracted & exported
    dataset_index = {}
    hash_table = {} # zobrist_hash: [{game_id}/{phase_name}]
    moves = {} # Moves frequency: {move: [n_no_press, n_press]}
    n_phases = OrderedDict() # n of phases per game
    end_supply_centers = {'press': {
                                power_name: {n_sc: [] for n_sc in range(0, supply_centers_to_win + 1)}
                                for power_name in all_powers},
                           'no_press': {
                               power_name: {n_sc: [] for n_sc in range(0, supply_centers_to_win + 1)}
                               for power_name in all_powers}}
    
    
    # opening files for writing extracted/processed data
    main_dataset = open(MODEL_DATA_PATHS["UNPROCESSED_DATASET_PATH"], "w")
    
    progress_bar = tqdm()
    
    # main data extraction loop
    with concurrent.futures.ProcessPoolExecutor(max_workers=DATA_EXTRACTION_NUM_WORKERS) as executor:
        for json_file_path in glob.glob(EXTRACTED_DATA_DIR + '/*.jsonl'):
            file_category = json_file_path.split('/')[-1].split('.')[0]
            dataset_index[file_category] = set()

            with open(json_file_path, 'r') as json_file:
                while True:

                    # reading in lines to the buffer
                    local_buffer = []
                    for i in range(DATA_EXTRACTION_BUFFER_SIZE):
                        local_buffer.append(json_file.readline())
                        
                    # checking if all buffer is empty (meaning we finished reading the file)
                    if(not any(local_buffer)):
                        break

                    for game_id, saved_game in executor.map(process_game, local_buffer):

                        if game_id is None:
                            continue

                        main_dataset.write(compress_game(saved_game) + "\n")

                        # Recording additional info
                        dataset_index[file_category].add(game_id)
                        end_supply_centers = get_end_scs_info(saved_game, game_id, all_powers, supply_centers_to_win, end_supply_centers)
                        moves = get_moves_info(saved_game, moves)
                        n_phases[game_id] = len(saved_game["phases"])

                        # Recording hash of each phase
                        for phase in saved_game["phases"]:
                            hash_table.setdefault(phase["state"]["zobrist_hash"], [])
                            hash_table[phase["state"]["zobrist_hash"]] += ['{}/{}'.format(game_id, phase["name"])]
                            
                        progress_bar.update(1)
                            
                            
    # Storing info to disk
    pickle.dump(dataset_index, open(MODEL_DATA_PATHS["DATASET_INDEX_PATH"], "wb"))
    pickle.dump(end_supply_centers, open(MODEL_DATA_PATHS["END_SCS_DATASET_PATH"], "wb"))
    pickle.dump(hash_table, open(MODEL_DATA_PATHS["HASH_DATASET_PATH"], "wb"))
    pickle.dump(moves, open(MODEL_DATA_PATHS["MOVES_COUNT_DATASET_PATH"], "wb"))
    pickle.dump(n_phases, open(MODEL_DATA_PATHS["PHASES_COUNT_DATASET_PATH"], "wb"))
    
    # closing files
    main_dataset.close()
    progress_bar.close()
    

if __name__ == "__main__":
    main()