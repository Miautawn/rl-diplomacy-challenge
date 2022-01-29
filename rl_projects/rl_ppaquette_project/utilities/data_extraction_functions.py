from diplomacy import Game, Map
import numpy as np

from utilities.utility_functions import (get_map_powers, get_top_victors,
                   get_sorted_locs, ALL_STANDARD_POWERS,
                   GO_ID, order_to_ix, get_board_alignments,
                   get_order_based_mask, get_issued_orders_for_powers,
                   get_possible_orders_for_powers, get_order_tokens, POWER_VOCABULARY_KEY_TO_IX,
                   get_current_season, compress_dict, decompress_dict)

from settings import (MODEL_DATA_PATHS, MODEL_DATA_DIR,
                    EXTRACTED_DATA_DIR, VALIDATION_SET_SPLIT,
                    N_LOCATIONS, N_SUPPLY_CENTERS,
                    N_LOCATION_FEATURES, N_ORDERS_FEATURES,
                    N_POWERS, N_SEASONS,
                    N_UNIT_TYPES, N_NODES,
                    TOKENS_PER_ORDER, MAX_LENGTH_ORDER_PREV_PHASES,
                    MAX_CANDIDATES, N_PREV_ORDERS, N_PREV_ORDERS_HISTORY)

def convert_to_board_representation(state, map_object):
    """ Converts a `.proto.game.State` proto to its matrix board state representation
        :param state: A `.proto.game.State` proto of the state of the game.
        :param map_object: The instantiated Map object
        :return: The board state (matrix representation) of the phase (81 x 35)
        :type map_object: diplomacy.Map
    """
    # Retrieving cached version directly from proto
    # if state.board_state:
    #     if len(state.board_state) == N_LOCATIONS * N_FEATURES:
    #         return np.array(state.board_state, dtype=np.uint8).reshape(N_NODES, NB_FEATURES)
    #     LOGGER.warning('Got a cached board state of dimension %d - Expected %d',
    #                    len(state.board_state), N_NODES * NB_FEATURES)

    # Otherwise, computing it
    
    #sorted locations (topologically if possible)
    locations = get_sorted_locs(map_object)
    
    #sorted supply centers of the map
    supply_centers = sorted([center.upper() for center in map_object.scs])
    
    #sorted powers of the map
    powers = get_map_powers(map_object)
    remaining_supply_centers = supply_centers.copy()

    # Sizes
    n_locations = len(locations)
    n_powers = len(powers)

    # Creating matrix components for locations according to the board representation matrix
    locations_unit_type_matrix = np.zeros((n_locations, N_UNIT_TYPES + 1), dtype=np.uint8)
    locations_unit_power_matrix = np.zeros((n_locations, n_powers + 1), dtype=np.uint8)
    locations_build_removable_matrix = np.zeros((n_locations, 2), dtype=np.uint8)
    locations_dislodged_unit_type_matrix = np.zeros((n_locations, N_UNIT_TYPES + 1), dtype=np.uint8)
    locations_dislodged_unit_power_matrix = np.zeros((n_locations, n_powers + 1), dtype=np.uint8)
    locations_area_type_matrix = np.zeros((n_locations, 3), dtype=np.uint8)
    locations_supply_center_owner = np.zeros((n_locations, n_powers + 1), dtype=np.uint8)

    # Settings units
    for power_name in state["units"].keys():
        build_count = state["builds"][power_name]["count"]

        # Marking regular and removable units
        for unit in state["units"][power_name]:
            # Checking in what phase we are in
            is_dislodged = bool(unit[0] == '*')

            # Removing leading * if dislodged
            unit = unit[1:] if is_dislodged else unit
            unit_type = unit[0]
            unit_location = unit[2:]
            unit_location_idx = locations.index(unit_location)

            # Calculating unit owner ix and unit type ix
            power_idx = powers.index(power_name)
            unit_type_idx = 0 if unit_type == 'A' else 1
            
            if not is_dislodged:
                locations_unit_power_matrix[unit_location_idx, power_idx] = 1
                locations_unit_type_matrix[unit_location_idx, unit_type_idx] = 1
            else:
                locations_dislodged_unit_power_matrix[unit_location_idx, power_idx] = 1
                locations_dislodged_unit_type_matrix[unit_location_idx, unit_type_idx] = 1

            # Setting number of removable units ???
            if build_count < 0:
                locations_build_removable_matrix[unit_location_idx, 1] = 1

            # Also setting the parent location if it's a coast
            if '/' in unit_location:
                location_without_coast = unit_location[:3]
                location_without_coast_idx = locations.index(location_without_coast)
                if not is_dislodged:
                    locations_unit_power_matrix[location_without_coast_idx, power_idx] = 1
                    locations_unit_type_matrix[location_without_coast_idx, unit_type_idx] = 1
                else:
                    locations_dislodged_unit_power_matrix[location_without_coast_idx, power_idx] = 1
                    locations_dislodged_unit_type_matrix[location_without_coast_idx, unit_type_idx] = 1
                if build_count < 0:
                    locations_build_removable_matrix[location_without_coast_idx, 1] = 1

        # Handling build locations
        if build_count > 0:
            buildable_locations = [loc for loc in locations if loc[:3] in state["builds"][power_name]["homes"]]

            # Marking location as buildable (with no units on it)
            for loc in buildable_locations:
                build_location_idx = locations.index(loc)

                # There are no units on it, so "Normal unit" is None
                locations_unit_type_matrix[build_location_idx, -1] = 1
                locations_unit_power_matrix[build_location_idx, -1] = 1
                locations_build_removable_matrix[build_location_idx, 0] = 1

    # Setting rows with no values to None
    locations_unit_type_matrix[(np.sum(locations_unit_type_matrix, axis=1) == 0, -1)] = 1
    locations_unit_power_matrix[(np.sum(locations_unit_power_matrix, axis=1) == 0, -1)] = 1
    locations_dislodged_unit_type_matrix[(np.sum(locations_dislodged_unit_type_matrix, axis=1) == 0, -1)] = 1
    locations_dislodged_unit_power_matrix[(np.sum(locations_dislodged_unit_power_matrix, axis=1) == 0, -1)] = 1

    # Setting area type
    for loc in locations:
        loc_idx = locations.index(loc)
        area_type = map_object.area_type(loc)
        if area_type in ['PORT', 'COAST']:
            area_type_idx = 2
        elif area_type == 'WATER':
            area_type_idx = 1
        elif area_type == 'LAND':
            area_type_idx = 0
        else:
            raise RuntimeError('Unknown area type {}'.format(area_type))
        locations_area_type_matrix[loc_idx, area_type_idx] = 1

    # Supply center ownership
    for power_name in state["centers"].keys():
        if power_name == 'UNOWNED':
            continue
        for center in state["centers"][power_name]:
            for loc in [map_loc for map_loc in locations if map_loc[:3] == center[:3]]:
                if loc[:3] in remaining_supply_centers:
                    remaining_supply_centers.remove(loc[:3])
                loc_idx = locations.index(loc)
                power_idx = powers.index(power_name)
                locations_supply_center_owner[loc_idx, power_idx] = 1

    # Unowned supply centers
    for center in remaining_supply_centers:
        for loc in [map_loc for map_loc in locations if map_loc[:3] == center[:3]]:
            loc_idx = locations.index(loc)
            
            #why cant this be just -1
            power_idx = n_powers
            locations_supply_center_owner[loc_idx, power_idx] = 1

    # Merging and returning
    return np.concatenate([locations_unit_type_matrix,
                           locations_unit_power_matrix,
                           locations_build_removable_matrix,
                           locations_dislodged_unit_type_matrix,
                           locations_dislodged_unit_power_matrix,
                           locations_area_type_matrix,
                           locations_supply_center_owner], axis=1)


def convert_to_previous_orders_representation(phase, map_object):
    """ Converts a `.proto.game.PhaseHistory` proto to its matrix prev orders state representation
        :param phase: A `.proto.game.PhaseHistory` proto of the phase history of the game.
        :param map_object: The instantiated Map object
        :return: The prev orders state (matrix representation) of the prev orders (81 x 40)
        :type map_object: diplomacy.Map
    """
    # Retrieving cached version directly from proto
    # if phase.prev_orders_state:
    #     if len(phase.prev_orders_state) == N_NODES * N_ORDERS_FEATURES:
    #         return np.array(phase.prev_orders_state, dtype=np.uint8).reshape(N_NODES, N_ORDERS_FEATURES)
    #     LOGGER.warning('Got a cached prev orders state of dimension %d - Expected %d',
    #                    len(phase.prev_orders_state), N_NODES * N_ORDERS_FEATURES)

    # Otherwise, computing it
    #sorted locations (topologically if possible)
    locations = get_sorted_locs(map_object)
    
    #sorted powers of the map
    powers = get_map_powers(map_object)
    
    #sorted supply centers of the map
    supply_centers = sorted([center.upper() for center in map_object.scs])

    # Sizes
    n_locations = len(locations)
    n_powers = len(powers)

    # Not a movement phase
    if phase["name"][-1] != 'M':
        LOGGER.warning('Trying to compute the prev_orders_state of a non-movement phase.')
        return np.zeros((N_LOCATIONS, N_FEATURES), dtype=np.uint8)

    # Creating matrix components for locations
    locations_unit_type_matrix = np.zeros((n_locations, N_UNIT_TYPES + 1), dtype=np.uint8)
    locations_issuing_power_matrix = np.zeros((n_locations, n_powers + 1), dtype=np.uint8)
    locations_order_type_matrix = np.zeros((n_locations, 5), dtype=np.uint8)
    locations_source_power_matrix = np.zeros((n_locations, n_powers + 1), dtype=np.uint8)
    locations_destination_power_matrix = np.zeros((n_locations, n_powers + 1), dtype=np.uint8)
    locations_supply_center_owner_matrix = np.zeros((n_locations, n_powers + 1), dtype=np.uint8)

    # Storing the owners of each location
    # The owner of a location is the unit owner if there is a unit, otherwise the supply_center owner, otherwise None
    location_owners = {}
    for power_name in phase["state"]["units"].keys():
        for unit in phase["state"]["units"][power_name]:
            loc = unit.split()[-1]
            location_owners[loc[:3]] = power_name

    # Storing the owners of each center
    remaining_supply_centers = supply_centers[:]
    for power_name in phase["state"]["centers"].keys():
        if power_name == 'UNOWNED':
            continue
        for center in phase["state"]["centers"][power_name]:
            for loc in [map_loc for map_loc in locations if map_loc[:3] == center[:3]]:
                if loc[:3] not in location_owners:
                    location_owners[loc[:3]] = power_name
                locations_supply_center_owner_matrix[locations.index(loc), powers.index(power_name)] = 1
            remaining_supply_centers.remove(center)
    for center in remaining_supply_centers:
        for loc in [map_loc for map_loc in locations if map_loc[:3] == center[:3]]:
            locations_supply_center_owner_matrix[locations.index(loc), -1] = 1

    # Parsing each order
    for issuing_power_name in phase["orders"].keys():
        issuing_power_index = powers.index(issuing_power_name)
        
        for order in phase["orders"][issuing_power_name] if phase["orders"][issuing_power_name] is not None else []:
            word = order.split()

            # Movement phase - Expecting Hold, Move, Support or Convoy
            if len(word) <= 2 or word[2] not in 'H-SC':
                LOGGER.warning('Unsupported order %s', order)
                continue

            # Detecting unit type, location and order type
            unit_type, unit_location, order_type = word[:3]
            unit_type_index = 0 if unit_type == 'A' else 1
            order_type_index = 'H-SC'.index(order_type)

            # Adding both with and without coasts
            unit_locations = [unit_location]
            if '/' in unit_location:
                unit_locations += [unit_location[:3]]

            for unit_loc in unit_locations:
                unit_location_index = locations.index(unit_loc)

                # Setting unit type, loc, order type
                locations_unit_type_matrix[unit_location_index, unit_type_index] = 1
                locations_issuing_power_matrix[unit_location_index, issuing_power_index] = 1
                locations_order_type_matrix[unit_location_index, order_type_index] = 1

                # Hold order
                if order_type == 'H':
                    locations_source_power_matrix[unit_location_index, -1] = 1
                    locations_destination_power_matrix[unit_location_index, -1] = 1

                # Move order
                elif order_type == '-':
                    destination = word[-1]
                    destination_power_index = -1 if destination[:3] not in location_owners else powers.index(location_owners[destination[:3]])
                    locations_source_power_matrix[unit_location_index, -1] = 1
                    locations_destination_power_matrix[unit_location_index, destination_power_index] = 1

                # Support hold
                elif order_type == 'S' and '-' not in word:
                    source = word[-1]
                    source_power_index = -1 if source[:3] not in location_owners else powers.index(location_owners[source[:3]])
                    locations_source_power_matrix[unit_location_index, source_power_index] = 1
                    locations_destination_power_matrix[unit_location_index, -1] = 1

                # Support move / Convoy
                elif order_type in ('S', 'C') and '-' in word:
                    source = word[word.index('-') - 1]
                    destination = word[-1]
                    source_power_index = -1 if source[:3] not in location_owners else powers.index(location_owners[source[:3]])
                    destination_power_index = -1 if destination[:3] not in location_owners else powers.index(location_owners[destination[:3]])
                    locations_source_power_matrix[unit_location_index, source_power_index] = 1
                    locations_destination_power_matrix[unit_location_index, destination_power_index] = 1

                # Unknown other
                else:
                    LOGGER.error('Unsupported order - %s', order)
                

    # Setting rows with no values to None
    locations_unit_type_matrix[(np.sum(locations_unit_type_matrix, axis=1) == 0, -1)] = 1
    locations_issuing_power_matrix[(np.sum(locations_issuing_power_matrix, axis=1) == 0, -1)] = 1
    locations_order_type_matrix[(np.sum(locations_order_type_matrix, axis=1) == 0, -1)] = 1
    locations_source_power_matrix[(np.sum(locations_source_power_matrix, axis=1) == 0, -1)] = 1
    locations_destination_power_matrix[(np.sum(locations_destination_power_matrix, axis=1) == 0, -1)] = 1

    # Adding prev order state at the beginning of the list (to keep the phases in the correct order)
    return np.concatenate([locations_unit_type_matrix,
                           locations_issuing_power_matrix,
                           locations_order_type_matrix,
                           locations_source_power_matrix,
                           locations_destination_power_matrix,
                           locations_supply_center_owner_matrix,], axis=1)


def add_cached_states_to_saved_game(saved_game):
    """ Adds a cached representation of board_state and prev_orders_state to the saved game """
    if saved_game['map'].startswith('standard'):
        map_object = Map(saved_game['map'])
        for idx, phase in enumerate(saved_game['phases']):
            phase['state']['board_state'] = convert_to_board_representation(phase['state'], map_object)
            if phase['name'][-1] == 'M':
                phase['previous_orders_state'] = convert_to_previous_orders_representation(phase, map_object)
            else:
                phase['previous_orders_state'] = np.array([])
    else:
        print("Not standard")
    return saved_game

def add_possible_orders_to_saved_game(saved_game):
    """ Adds possible_orders for each phase of the saved game """
    if saved_game['map'].startswith('standard'):
        for phase in saved_game['phases']:
            game = Game(map_name=saved_game['map'], rules=saved_game['rules'])
            game.set_state(phase['state'])
            phase['possible_orders'] = game.get_all_possible_orders()
    return saved_game

def add_rewards_to_saved_game(saved_game, reward_fn):
    """ Adds a cached list of rewards for each power in the game according the the reward fn """
    if reward_fn is not None and saved_game["map"].startswith('standard'):
        
        if saved_game.get("reward_fn") is None:
            saved_game["reward_fn"] = reward_fn.name
            
        if saved_game.get("rewards") is None:
            saved_game["rewards"] = {}
            
        if saved_game.get("done_reason") is None:
            saved_game["done_reason"] = ''

        for power_name in ALL_STANDARD_POWERS:
            power_rewards = reward_fn.get_episode_rewards(saved_game, power_name)
            
            if saved_game["rewards"].get(power_name) is None:
                saved_game["rewards"][power_name] = power_rewards
            else:
                saved_game["rewards"][power_name].extend(power_rewards)
    return saved_game

def get_end_scs_info(saved_game, game_id, all_powers, supply_centers_to_win, end_supply_centers):
    """ Records the ending supply center information """
    # Only keeping games with the standard map (or its variations) that don't use BUILD_ANY
    if not saved_game["map"].startswith('standard'):
        return
    if 'BUILD_ANY' in saved_game["rules"]:
        return

    # Detecting if no press
    is_no_press = 'NO_PRESS' in saved_game["rules"]

    # Counting the nunmber of ending scs for each power
    for power_name in all_powers:
        n_of_supply_centers = min(supply_centers_to_win, len(saved_game["phases"][-1]["state"]["centers"].get(power_name, [])))
        if is_no_press:
            end_supply_centers['no_press'][power_name][n_of_supply_centers] += [game_id]
        else:
            end_supply_centers['press'][power_name][n_of_supply_centers] += [game_id]
            
    return end_supply_centers

def get_moves_info(saved_game, moves):
    """ Recording the frequency of each order """
    # Only keeping games with the standard map (or its variations)
    if not saved_game["map"].startswith('standard'):
        return

    # Detecting if no press
    is_no_press = 'NO_PRESS' in saved_game["rules"]

    # Counting all orders
    for phase in saved_game["phases"]:
        for power_name in phase["orders"].keys():
            for order in phase["orders"][power_name] if phase["orders"][power_name] is not None else []:
                moves.setdefault(order, [0, 0])
                moves[order][is_no_press] += 1
                
    return moves

def compress_game(game_object):
    """
    Convert a game object to compressed string
    """
    
    # convert np.arrays to list in order to serialize them
    for idx, phase in enumerate(game_object["phases"]):
        if(type(phase["state"]["board_state"]) != list):
            game_object["phases"][idx]["state"]["board_state"] = phase["state"]["board_state"].tolist()

        if(type(phase["previous_orders_state"]) != list):
            game_object["phases"][idx]["previous_orders_state"] = phase["previous_orders_state"].tolist()

    #returns encoded game bytes as a string
    encoded_game = compress_dict(game_object)
    return str(encoded_game)


def decompress_game(game_bytes_string, convertable_columns = []):
    """
    Convert a compressed string back to game object
    """
    game = decompress_dict(game_bytes_string, convertable_columns)

    #for faster use, convert lists to np.arrays
    for idx, phase in enumerate(game["phases"]):
        game["phases"][idx]["state"]["board_state"] = np.array(phase["state"]["board_state"])
        game["phases"][idx]["previous_orders_state"] = np.array(phase["previous_orders_state"])

    return game
            
            





    