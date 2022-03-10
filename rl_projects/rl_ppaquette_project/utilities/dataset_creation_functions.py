import logging
import pickle

import numpy as np
from diplomacy import Game, Map

from utilities.utility_functions import (get_map_powers, get_top_victors,
                   get_sorted_locs, ALL_STANDARD_POWERS,
                   GO_ID, order_to_ix, get_board_alignments,
                   get_order_based_mask, get_issued_orders_for_powers,
                   get_possible_orders_for_powers, get_order_tokens, POWER_VOCABULARY_KEY_TO_IX,
                   get_current_season)

from utilities.data_extraction_functions import (convert_to_board_representation,
                                                convert_to_previous_orders_representation,
                                                add_cached_states_to_saved_game,
                                                add_possible_orders_to_saved_game,
                                                add_rewards_to_saved_game,
                                                get_end_scs_info, get_moves_info,
                                                compress_game, decompress_game)

from settings import (MODEL_DATA_PATHS, MODEL_DATA_DIR,
EXTRACTED_DATA_DIR, VALIDATION_SET_SPLIT,
N_LOCATIONS, N_SUPPLY_CENTERS,
N_LOCATION_FEATURES, N_ORDERS_FEATURES,
N_POWERS, N_SEASONS,
N_UNIT_TYPES, N_NODES,
TOKENS_PER_ORDER, MAX_LENGTH_ORDER_PREV_PHASES,
MAX_CANDIDATES, N_PREV_ORDERS, N_PREV_ORDERS_HISTORY, DATA_BLUEPRINT)


LOGGER = logging.getLogger(__name__)

def get_request_id(saved_game, phase_idx, power_name, is_validation_set):
    """ Returns the standardized request id for this game/phase """
    return '%s/%s/%d/%s/%s' % ('train' if not is_validation_set else 'valid',
                               saved_game["id"],
                               phase_idx,
                               saved_game["phases"][phase_idx]["name"],
                               power_name)


def get_policy_data(saved_game, power_names, top_victors):
    """ Computes the proto to save in tf.train.Example as a training example for the policy network
        :param saved_game: A `.proto.game.SavedGame` object from the dataset.
        :param power_names: The list of powers for which we want the policy data
        :param top_victors: The list of powers that ended with more than 25% of the supply centers
        :return: A dictionary with key: the phase_idx
                              with value: A dict with the power_name as key and a dict with the example fields as value
    """
    n_phases = len(saved_game["phases"])
    policy_data = {phase_idx: {} for phase_idx in range(n_phases - 1)}
    game_id = saved_game["id"]
    map_object = Map(saved_game["map"])

    # Determining if we have a draw
    n_sc_to_win = len(map_object.scs) // 2 + 1
    has_solo_winner = max([len(saved_game["phases"][-1]["state"]["centers"][power_name])
                           for power_name in saved_game["phases"][-1]["state"]["centers"]]) >= n_sc_to_win
    survivors = [power_name for power_name in saved_game["phases"][-1]["state"]["centers"]
                 if saved_game["phases"][-1]["state"]["centers"][power_name]]
    has_draw = not has_solo_winner and len(survivors) >= 2

    # Processing all phases (except the last one)
    current_year = 0
    for phase_idx in range(n_phases - 1):

        # Building a list of orders of previous phases
        previous_orders_states = [np.zeros((N_NODES, N_ORDERS_FEATURES), dtype=np.uint8)] * N_PREV_ORDERS
        for phase in saved_game["phases"][max(0, phase_idx - N_PREV_ORDERS_HISTORY):phase_idx]:
            if phase["name"][-1] == 'M':
                previous_orders_states += [convert_to_previous_orders_representation(phase, map_object)]
        previous_orders_states = previous_orders_states[-N_PREV_ORDERS:]
        prev_orders_state = np.array(previous_orders_states)

        # Parsing each requested power in the specified phase
        phase = saved_game["phases"][phase_idx]
        phase_name = phase["name"]
        state = phase["state"]
        phase_board_state = convert_to_board_representation(state, map_object)

        # Increasing year for every spring or when the game is completed
        if phase["name"] == 'COMPLETED' or (phase["name"][0] == 'S' and phase["name"][-1] == 'M'):
            current_year += 1

        for power_name in power_names:
            phase_issued_orders = get_issued_orders_for_powers(phase, [power_name])
            phase_possible_orders = get_possible_orders_for_powers(phase, [power_name])
            phase_draw_target = 1. if has_draw and phase_idx == (n_phases - 2) and power_name in survivors else 0.

            # Data to use when not learning a policy
            blank_policy_data = {'board_state': phase_board_state,
                                 'prev_orders_state': prev_orders_state,
                                 'draw_target': phase_draw_target}

            # Power is not a top victor - We don't want to learn a policy from him
            if power_name not in top_victors:
                policy_data[phase_idx][power_name] = blank_policy_data
                continue

            # Finding the orderable locs
            orderable_locations = list(phase_issued_orders[power_name].keys())

            # Skipping power for this phase if we are only issuing Hold
            for order_loc, order in phase_issued_orders[power_name].items():
                order_tokens = get_order_tokens(order)
                if len(order_tokens) >= 2 and order_tokens[1] != 'H':
                    break
            else:
                policy_data[phase_idx][power_name] = blank_policy_data
                continue

            # Removing orderable locs where orders are not possible (i.e. NO_CHECK games)
            for order_loc, order in phase_issued_orders[power_name].items():
                if order not in phase_possible_orders[order_loc] and order_loc in orderable_locations:
                    if 'NO_CHECK' not in saved_game["rules"]:
                        LOGGER.warning('%s not in all possible orders. Phase %s - Game %s.', order, phase_name, game_id)
                    orderable_locations.remove(order_loc)

                # Remove orderable locs where the order is either invalid or not frequent
                if order_to_ix(order) is None and order_loc in orderable_locations:
                    orderable_locations.remove(order_loc)

            # Determining if we are in an adjustment phase
            in_adjustment_phase = state["name"][-1] == 'A'
            n_builds = state["builds"][power_name]["count"]
            n_homes = len(state["builds"][power_name]["homes"])

            # WxxxA - We can build units
            # WxxxA - We can disband units
            # Other phase
            if in_adjustment_phase and n_builds >= 0:
                decoder_length = min(n_builds, n_homes)
            elif in_adjustment_phase and n_builds < 0:
                decoder_length = abs(n_builds)
            else:
                decoder_length = len(orderable_locations)

            # Not all units were disbanded - Skipping this power as we can't learn the orders properly
            if in_adjustment_phase and n_builds < 0 and len(orderable_locations) < abs(n_builds):
                policy_data[phase_idx][power_name] = blank_policy_data
                continue

            # Not enough orderable locations for this power, skipping
            if not orderable_locations or not decoder_length:
                policy_data[phase_idx][power_name] = blank_policy_data
                continue

            # decoder_inputs [GO, order1, order2, order3]
            decoder_inputs = [GO_ID]
            decoder_inputs += [order_to_ix(phase_issued_orders[power_name][loc]) for loc in orderable_locations]
            if in_adjustment_phase and n_builds > 0:
                decoder_inputs += [order_to_ix('WAIVE')] * (min(n_builds, n_homes) - len(orderable_locations))
            decoder_length = min(decoder_length, N_SUPPLY_CENTERS)

            # Adjustment Phase - Use all possible orders for each location.
            if in_adjustment_phase:
                build_disband_locs = list(get_possible_orders_for_powers(phase, [power_name]).keys())
                phase_board_alignments = get_board_alignments(build_disband_locs,
                                                              in_adjustment_phase=in_adjustment_phase,
                                                              tokens_per_loc=1,
                                                              decoder_length=decoder_length)

                # Building a list of all orders for all locations
                adj_orders = []
                for loc in build_disband_locs:
                    adj_orders += phase_possible_orders[loc]

                # Not learning builds for BUILD_ANY
                if n_builds > 0 and 'BUILD_ANY' in state["rules"]:
                    adj_orders = []

                # No orders found - Skipping
                if not adj_orders:
                    policy_data[phase_idx][power_name] = blank_policy_data
                    continue

                # Computing the candidates
                candidates = [get_order_based_mask(adj_orders)] * decoder_length

            # Regular phase - Compute candidates for each issued order location 
            else:
                phase_board_alignments = get_board_alignments(orderable_locations,
                                                              in_adjustment_phase=in_adjustment_phase,
                                                              tokens_per_loc=1,
                                                              decoder_length=decoder_length)
                
                candidates = []
                for loc in orderable_locations:
                    candidates += [get_order_based_mask(phase_possible_orders[loc])]

            # Saving results
            # No need to return temperature, current_power, current_season
           
            policy_data[phase_idx][power_name] = {'board_state': phase_board_state,
                                                 'board_alignments': phase_board_alignments,
                                                 'prev_orders_state': prev_orders_state,
                                                 'decoder_inputs': decoder_inputs,
                                                 'decoder_lengths': decoder_length,
                                                 'candidates': candidates,
                                                 'draw_target': phase_draw_target}
    return policy_data


def format_datapoint(data):
    """
    Acording to the data blueprint:
        1.) Add missing fields
        2.) Pad the requered fields
    """
    
    
    for feature_name, feature_structure in DATA_BLUEPRINT.items():
        
        # Adding missing fields
        if feature_name not in data:
            if feature_structure["type"] == "static":
                data[feature_name] = np.zeros(feature_structure["shape"])
            elif feature_structure["type"] == "variable":
                data[feature_name] = []
                
        #padding the static features
        if feature_structure["type"] == "static" and feature_structure["shape"]:
            data_type = type(data[feature_name])
            padded_feature = np.array(data[feature_name])
            padded_feature = np.resize(padded_feature, feature_structure["shape"])
            data[feature_name] = padded_feature.tolist() if data_type == list else padded_feature
            
        ## EXCEPTION!!! REPLACE!!
        ## In the original code they convert numpy arrays to bytes, which 
        ## does not retain structure information, thus causes flattening on "board alignments" feature
        ## Here we do it artificially
        if feature_name == "board_alignments":
            data[feature_name] = sum(data[feature_name], [])
            
    return data


def generate_datapoint(saved_game, is_validation_set):

    """ Converts a dataset game to protocol buffer format
        :param saved_game_bytes: A `.proto.game.SavedGame` object from the dataset.
        :param is_validation_set: Boolean that indicates if we are generating the validation set (otw. training set)
        :return: A dictionary with phase_idx as key and a dictionary {power_name: (msg_len, proto)} as value
    """
    
    if not saved_game["map"].startswith('standard'):
        return {phase_idx: [] for phase_idx, _ in enumerate(saved_game["phases"])}

    # Finding top victors and supply centers at end of game
    map_object = Map(saved_game["map"])
    top_victors = get_top_victors(saved_game, map_object)
    all_powers = get_map_powers(map_object)
    n_phases = len(saved_game["phases"])
    game_results = {phase_idx: {} for phase_idx in range(n_phases)}

    # Getting policy data for the phase_idx
    # (All powers for training - Top victors for evaluation)
    policy_data = get_policy_data(saved_game,
                                  power_names=all_powers,
                                  top_victors=top_victors if is_validation_set else all_powers)

    # Building results
    for phase_idx in range(n_phases - 1):
        for power_name in all_powers:
            if is_validation_set and power_name not in top_victors:
                continue
            phase_policy = policy_data[phase_idx][power_name]

            if 'decoder_inputs' not in phase_policy:
                continue

            request_id = get_request_id(saved_game, phase_idx, power_name, is_validation_set)
            data = {'request_id': request_id,
                    'player_seed': 0,
                    'decoder_inputs': [GO_ID],
                    'noise': 0.,
                    'temperature': 0.,
                    'dropout_rate': 0.,
                    'current_power': POWER_VOCABULARY_KEY_TO_IX[power_name],
                    'current_season': get_current_season(saved_game["phases"][phase_idx]["state"])}
            data.update(phase_policy)

            # Saving results
            game_result = format_datapoint(data)
            game_results[phase_idx][power_name] = (0, game_result)

    # Returning data for buffer
    return game_results
