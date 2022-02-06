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

from utilities.data_extraction_functions import compress_game, decompress_game
                                                 
from utilities.utility_functions import compress_dict, decompress_dict

from settings import (MODEL_DATA_PATHS, MODEL_DATA_DIR,
EXTRACTED_DATA_DIR, VALIDATION_SET_SPLIT,
N_LOCATIONS, N_SUPPLY_CENTERS,
N_LOCATION_FEATURES, N_ORDERS_FEATURES,
N_POWERS, N_SEASONS,
N_UNIT_TYPES, N_NODES,
TOKENS_PER_ORDER, MAX_LENGTH_ORDER_PREV_PHASES,
MAX_CANDIDATES, N_PREV_ORDERS, N_PREV_ORDERS_HISTORY)


# ---------- Multiprocessing method for generating datasets ----------------
def handle_queues(task_ids, callable_fn, saved_game, is_validation_set):
    """ Handles the queued datapoint processing"""
    try:
        game_results = callable_fn(saved_game, is_validation_set)
        results = []
        for phase_idx, task_id in enumerate(task_ids):
            if not game_results[phase_idx]:
                results.append((task_id, None, 0, None))
                continue
            for power_name in game_results[phase_idx]:
                message_lengths, game_result = game_results[phase_idx][power_name]
                results.append((task_id, power_name, message_lengths, game_result))
        return results
    except Exception as exc:
        raise exc


def main():
    
    ############
    ## The code below generates train & validation datasets
    ############
    
    LOGGER = logging.getLogger()

    # Loading index
    dataset_index = {}

    # Splitting games into training and validation
    phase_count_dataset = pickle.load(open(MODEL_DATA_PATHS["PHASES_COUNT_DATASET_PATH"], 'rb'))      
    game_ids = list(phase_count_dataset.keys())

    n_valid = int(VALIDATION_SET_SPLIT * len(game_ids))
    set_valid_ids = set(sorted(game_ids)[-n_valid:])
    train_ids = list(sorted([game_id for game_id in game_ids if game_id not in set_valid_ids]))
    valid_ids = list(sorted(set_valid_ids))

    # Building a list of all task phases
    train_task_phases, valid_task_phases = [], []

    for game_id in train_ids:
        for phase_idx in range(phase_count_dataset.get(game_id, 0)):
            train_task_phases += [(phase_idx, game_id)]
    for game_id in valid_ids:
        for phase_idx in range(phase_count_dataset.get(game_id, 0)):
            valid_task_phases += [(phase_idx, game_id)]


    # Grouping tasks by buckets
    # buckets_pending contains a set of pending task ids in each bucket
    # buckets_keys contains a list of tuples so we can group items with similar message length together
    task_to_bucket = {}                                             # {task_id: bucket_id}
    train_buckets_pending, valid_buckets_pending = [], []           # [bucket_id: set()]
    train_buckets_keys, valid_buckets_keys = [], []                 # [bucket_id: (msg_len, task_id, power_name)]


    # Building a dictionary of {game_id: {phase_idx: task_id}}
    # Train tasks have a id >= 0, valid tasks have an id < 0
    task_id = 1
    task_id_per_game = {}
    for phase_idx, game_id in train_task_phases:
        task_id_per_game.setdefault(game_id, {})[phase_idx] = task_id
        task_id += 1

    task_id = -1
    for phase_idx, game_id in valid_task_phases:
        task_id_per_game.setdefault(game_id, {})[phase_idx] = task_id
        task_id -= 1
        
    
    # Building a dictionary of pending items, so we can write them to disk in the correct order
    n_train_tasks = len(train_task_phases)
    n_valid_tasks = len(valid_task_phases)
    pending_train_tasks = OrderedDict({task_id: None for task_id in range(1, n_train_tasks + 1)})
    pending_valid_tasks = OrderedDict({task_id: None for task_id in range(1, n_valid_tasks + 1)})

    # Computing batch_size, progress bar and creating a pool of processes
    batch_size = 5120
    progress_bar = tqdm(total=n_train_tasks + n_valid_tasks)
    process_pool = concurrent.futures.ProcessPoolExecutor()
    futures = set()

    # Creating buffer to write all protos to disk at once
    train_buffer, valid_buffer = Queue(), Queue()

    # Opening the proto file to read games
    full_dataset = open(MODEL_DATA_PATHS["UNPROCESSED_DATASET_PATH"], "r")
    n_items_being_processed = 0

    # Creating training and validation dataset
    for training_mode in ['train', 'valid']:
        next_key = 1
        current_bucket = 0

        if training_mode == 'train':
            pending_tasks = pending_train_tasks
            buckets_pending = train_buckets_pending
            buckets_keys = train_buckets_keys
            buffer = train_buffer
            max_next_key = n_train_tasks + 1
        else:
            pending_tasks = pending_valid_tasks
            buckets_pending = valid_buckets_pending
            buckets_keys = valid_buckets_keys
            buffer = valid_buffer
            max_next_key = n_valid_tasks + 1
            
        dataset_index['size_{}_dataset'.format(training_mode)] = 0
        
        # Processing with a queue to avoid high memory usage
        while pending_tasks:

            # Filling queues
            while batch_size > n_items_being_processed:
                saved_game_string = full_dataset.readline()
                if not saved_game_string or saved_game_string in [None, "\n"]:
                    break
                    
                saved_game = decompress_game(saved_game_string)
                game_id = saved_game["id"]
                if game_id not in task_id_per_game:
                    continue
                n_phases = len(saved_game["phases"])
                task_ids = [task_id_per_game[game_id][phase_idx] for phase_idx in range(n_phases)]
                futures.add((tuple(task_ids), process_pool.submit(handle_queues,
                                                                  task_ids,
                                                                  generate_datapoint,
                                                                  saved_game,
                                                                  task_ids[0] < 0)))
                n_items_being_processed += n_phases

            # Processing results
            for expected_task_ids, future in list(futures):
                if not future.done():
                    continue
                results = future.result()
                current_task_ids = set()

                # Storing in compressed format in memory
                for task_id, power_name, message_lengths, game_result in results:
                    current_task_ids.add(task_id)

                    if game_result is not None:
                        # zlib_result = proto_to_zlib(game_result)
                        if task_id > 0:
                            if pending_train_tasks[abs(task_id)] is None:
                                pending_train_tasks[abs(task_id)] = {}
                            pending_train_tasks[abs(task_id)][power_name] = game_result
                        else:
                            if pending_valid_tasks[abs(task_id)] is None:
                                pending_valid_tasks[abs(task_id)] = {}
                            pending_valid_tasks[abs(task_id)][power_name] = game_result

                    # No results - Marking task id as done
                    elif task_id > 0 and pending_train_tasks[abs(task_id)] is None:
                        del pending_train_tasks[abs(task_id)]
                    elif task_id < 0 and pending_valid_tasks[abs(task_id)] is None:
                        del pending_valid_tasks[abs(task_id)]

                # Missing some task ids
                if set(expected_task_ids) != current_task_ids:
                    LOGGER.warning('Missing tasks ids. Got %s - Expected: %s', current_task_ids, expected_task_ids)
                    current_task_ids = expected_task_ids

                # Marking tasks as completed
                n_items_being_processed -= len(expected_task_ids)
                progress_bar.update(len(current_task_ids))

                # Deleting futures to release memory
                futures.remove((expected_task_ids, future))
                del future

            # Writing to disk
            while True:

                # Writing to buffer in the same order as they are received
                if next_key >= max_next_key:
                    break
                if next_key not in pending_tasks:
                    next_key += 1
                    continue
                if pending_tasks[next_key] is None:
                    break
                zlib_results = pending_tasks.pop(next_key)
                for zlib_result in zlib_results.values():
                    buffer.put(zlib_result)
                    dataset_index['size_{}_dataset'.format(training_mode)] += 1
                next_key += 1
                del zlib_results
                
    # Stopping pool, and progress bar
    process_pool.shutdown(wait=True)
    progress_bar.close()
    full_dataset.close()

    # Storing protos to disk
    LOGGER.info('Writing datasets to disk...')
    progress_bar = tqdm(total=train_buffer.qsize() + valid_buffer.qsize())

    with open(MODEL_DATA_PATHS["TRAINING_DATASET_PATH"], "w") as training_data_writer:
        while not train_buffer.empty():
            game_result = train_buffer.get()
            training_data_writer.write(compress_dict(game_result) + "\n")
            progress_bar.update(1)

    with open(MODEL_DATA_PATHS["VALIDATION_DATASET_PATH"], "w") as validation_data_writer:
        while not valid_buffer.empty():
            game_result = valid_buffer.get()
            validation_data_writer.write(compress_dict(game_result) + "\n")
            progress_bar.update(1)

    pickle.dump(dataset_index, open(MODEL_DATA_PATHS["INDEX_DATASET_PATH"], "wb"))

    # Closing
    progress_bar.close()
    


if __name__ == "__main__":
    main()