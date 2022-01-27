from operator import itemgetter
from collections import OrderedDict
import zlib
import json

import numpy as np

from diplomacy import Game, Map
from settings import (N_LOCATIONS, N_SUPPLY_CENTERS,
N_LOCATION_FEATURES, N_ORDERS_FEATURES,
N_POWERS, N_SEASONS, N_UNIT_TYPES, N_NODES,
TOKENS_PER_ORDER, MAX_LENGTH_ORDER_PREV_PHASES,
MAX_CANDIDATES, N_PREV_ORDERS, N_PREV_ORDERS_HISTORY)

ALL_STANDARD_POWERS = ['AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY', 'ITALY', 'RUSSIA', 'TURKEY']

# Predefined location order
STANDARD_TOPOLOGICAL_LOCATIONS  = ['YOR', 'EDI', 'LON', 'LVP', 'NTH', 'WAL', 'CLY',
                                   'NWG', 'ENG', 'IRI', 'NAO', 'BEL', 'DEN', 'HEL',
                                   'HOL', 'NWY', 'SKA', 'BAR', 'BRE', 'MAO', 'PIC',
                                   'BUR', 'RUH', 'BAL', 'KIE', 'SWE', 'FIN', 'STP',
                                   'STP/NC', 'GAS', 'PAR', 'NAF', 'POR', 'SPA', 'SPA/NC',
                                   'SPA/SC', 'WES', 'MAR', 'MUN', 'BER', 'BOT', 'LVN',
                                   'PRU', 'STP/SC', 'MOS', 'TUN', 'LYO', 'TYS', 'PIE',
                                   'BOH', 'SIL', 'TYR', 'WAR', 'SEV', 'UKR', 'ION',
                                   'TUS', 'NAP', 'ROM', 'VEN', 'GAL', 'VIE', 'TRI',
                                   'ARM', 'BLA', 'RUM', 'ADR', 'AEG', 'ALB', 'APU',
                                   'EAS', 'GRE', 'BUD', 'SER', 'ANK', 'SMY', 'SYR',
                                   'BUL', 'BUL/EC', 'CON', 'BUL/SC']


# Cache for adjacency matrix and sorted locs
ADJACENCY_MATRIX = {}
SORTED_LOCATIONS = {}

def compress_dict(dict_object):
    
    # convert each np.ndarray to list within the dict    
    def deep_search(converted_dict):     
        for key, value in converted_dict.items():
            if isinstance(value, dict):
                compress_dict(value)
            if isinstance(value, np.ndarray):
                converted_dict[key] = value.tolist()

        return converted_dict

    dict_object = deep_search(dict_object)
    
    #returns encoded game bytes as a string
    encoded_dict = zlib.compress(json.dumps(dict_object).encode("utf-8"), level=-1)
    return str(encoded_dict)   


def decompress_dict(dict_bytes_string, convertable_columns = []):
    
    #turn from string-bytes to actual bytes
    dict_bytes = eval(dict_bytes_string)

    dict_object = json.loads(zlib.decompress(dict_bytes).decode("utf-8"))
    
    # convert each np.ndarray to list within the dict  
    def deep_search(converted_dict):
        for key, value in converted_dict.items():
            if isinstance(value, dict):
                deep_search(value)
            if isinstance(value, list) and key in convertable_columns:
                converted_dict[key] = np.array(value)

        return dict_object
    
    dict_object = deep_search(dict_object)
    
    return dict_object


def get_map_powers(map_object):
    """ Returns the list of powers on the map """
    if map_object.name.startswith('standard'):
        return ALL_STANDARD_POWERS
    return sorted([power_name for power_name in map_object.powers])

def get_sorted_locs(map_object):
    """ Returns the list of locations for the given map in sorted order, using topological order
        :param map_object: The instantiated map
        :return: A sorted list of locations
    """
    if map_object.name not in SORTED_LOCATIONS:
        
        #get the location sorting key (if standard then use the predefined order)
        sorting_key = None if not map_object.name.startswith('standard') else STANDARD_TOPOLOGICAL_LOCATIONS.index
        locations = [loc.upper() for loc in map_object.locs if map_object.area_type(loc) != 'SHUT']
        
        #add the sorted locations of a mad to the dict 
        SORTED_LOCATIONS[map_object.name] = sorted(locations, key=sorting_key)
    return SORTED_LOCATIONS[map_object.name]

def get_current_season(state):
    """ Returns the index of the current season (0 = S, 1 = F, 2 = W)
        :param state: A `.proto.game.State` object.
        :return: The integer representation of the current season
    """
    season = state["name"]
    if season == 'COMPLETED':
        return 0
    if season[0] not in 'SFW':
        LOGGER.warning('Unrecognized season %s. Using "Spring" as the current season.', season)
        return 0
    return 'SFW'.index(season[0])

def build_game_from_state(state):
    """ Builds a game object from a state """
    game = Game(map_name=state["map"], rules=state["rules"])
    game.set_current_phase(state["name"])

    # Setting units
    game.clear_units()
    for power_name in state["units"]:
        game.set_units(power_name, state["units"][power_name])

    # Setting centers
    game.clear_centers()
    for power_name in state["centers"]:
        game.set_centers(power_name, state["centers"][power_name])

    # Returning
    return game

def get_orders_by_loc(phase, orderable_locations, powers):
    """ Returns a dictionary with loc as key and its corresponding order as value
        Note: only the locs in orderable_locations are included in the dictionary

        :param phase: A `.proto.game.SavedGame.Phase` object from the dataset.
        :param orderable_locations: A list of locs from which we want the orders
        :param powers: A list of powers for which to retrieve orders
        :return: A dictionary with locs as key, and their corresponding order as value
               (e.g. {'PAR': 'A PAR - MAR'})
    """
    orders_by_loc = {}
    for power_name in phase["orders"]:
        if power_name not in powers:
            continue
        for order in phase["orders"][power_name]:
            order_loc = order.split()[1]

            # Skipping if not one of the orderable locations
            if order_loc not in orderable_locations and order_loc[:3] not in orderable_locations:
                continue

            # Adding order to dictionary
            # Removing coast from key
            orders_by_loc[order_loc[:3]] = order

    # Returning order by location
    return orders_by_loc

def get_top_victors(saved_game, map_object):
    """ Returns a list of the top victors (i.e. owning more than 25% -1 of the centers on the map)
        We will only used the orders from these victors for the supervised learning
        :param saved_game: A `.proto.game.SavedGame` object from the dataset.
        :param map_object: The instantiated Map object
        :return: A list of victors (powers)
        :type map_object: diplomacy.Map
    """
    powers = get_map_powers(map_object)
    n_scs = len(map_object.scs)
    min_n_scs = n_scs // 4 - 1
    

    # Retrieving the number of centers for each power at the end of the game
    # Only keeping powers with at least 7 centers
    scs_last_phase = saved_game["phases"][-1]["state"]["centers"]

    ending_scs = [(power_name, len(scs_last_phase.get(power_name, []))) for power_name in powers
                  if len(scs_last_phase.get(power_name, [])) >= min_n_scs]
    ending_scs = sorted(ending_scs, key=itemgetter(1), reverse=True)

    # Not victors found, returning all powers because they all are of similar strength
    if not ending_scs:
        return powers
    return [power_name for power_name, _ in ending_scs]

def get_orderable_locs_for_powers(state, powers, shuffled=False):
    """ Returns a list of all orderable locations and a list of orderable location for each power
        :param state: A `.proto.game.State` object.
        :param powers: A list of powers for which to retrieve the orderable locations
        :param shuffled: Boolean. If true, the orderable locations for each power will be shuffled.
        :return: A tuple consisting of:
            - A list of all orderable locations for all powers
            - A dictionary with the power name as key, and a set of its orderable locations as value
    """
    # Detecting if we are in retreats phase
    in_retreats_phase = state["name"][-1] == 'R'

    # Detecting orderable locations for each top victor
    # Not storing coasts for orderable locations
    all_orderable_locs = set()
    orderable_locs = {power_name: set() for power_name in powers}
    for power_name in powers:

        # Adding build locations
        if state["name"][-1] == 'A' and state["builds"].get(power_name, {'count': 0, 'homes': []})["count"] >= 0:
            for home in state["builds"].get(power_name, {'count': 0, 'homes': []})["homes"]:
                all_orderable_locs.add(home[:3])
                orderable_locs[power_name].add(home[:3])

        # Otherwise, adding units (regular and dislodged)
        else:
            for unit in state["units"].get(power_name, []):
                unit_type, unit_loc = unit.split()
                if unit_type[0] == '*' and in_retreats_phase:
                    all_orderable_locs.add(unit_loc[:3])
                    orderable_locs[power_name].add(unit_loc[:3])
                elif unit_type[0] != '*' and not in_retreats_phase:
                    all_orderable_locs.add(unit_loc[:3])
                    orderable_locs[power_name].add(unit_loc[:3])


    # Sorting orderable locations
    key = None if not state["map"].startswith('standard') else STANDARD_TOPOLOGICAL_LOCATIONS.index
    orderable_locs = {power_name: sorted(orderable_locs[power_name], key=key) for power_name in powers}

    # Shuffling list if requested
    if shuffled:
        for power_name in powers:
            shuffle(orderable_locs[power_name])

    # Returning
    return list(sorted(all_orderable_locs, key=key)), orderable_locs

def get_issued_orders_for_powers(phase, powers, shuffled=False):
    """ Extracts a list of orderable locations and corresponding orders for a list of powers
        :param phase: A `.proto.game.SavedGame.Phase` object from the dataset.
        :param powers: A list of powers for which we want issued orders
        :param shuffled: Boolean. If true, orderable locations are shuffled, otherwise they are sorted alphabetically
        :return: A dictionary with the power name as key, and for value a dictionary of:
                    - orderable location for that power as key (e.g. 'PAR')
                    - the corresponding order at that location as value (e.g. 'A PAR H')
    """
    # Retrieving orderable locations
    all_orderable_locs, orderable_locs = get_orderable_locs_for_powers(phase["state"],
                                                                       powers,
                                                                       shuffled=shuffled)

    # Retrieving orders by loc for orderable locations
    orders_by_loc = get_orders_by_loc(phase, all_orderable_locs, powers)

    # Computing list of issued orders for each top victor
    issued_orders = OrderedDict()
    for power_name in powers:
        issued_orders[power_name] = OrderedDict()
        for loc in orderable_locs[power_name]:
            for issued_order_loc in [order_loc for order_loc in orders_by_loc
                                     if order_loc == loc or (order_loc[:3] == loc[:3] and '/' in order_loc)]:
                issued_orders[power_name][issued_order_loc] = orders_by_loc[issued_order_loc]

    # Returning
    return issued_orders

def get_possible_orders_for_powers(phase, powers):
    """ Extracts a list of possible orders for all locations where a power could issue an order
        :param phase: A `.proto.game.SavedGame.Phase` object from the dataset.
        :param powers: A list of powers for which we want the possible orders
        :return: A dictionary for each location, the list of possible orders
    """
    possible_orders_original = phase["possible_orders"]

    # Making sure we have a list of possible orders attached to the phase
    # Otherwise, creating a game object to retrieve the possible orders
    if not possible_orders_original:
        LOGGER.warning('The list of possible orders was not attached to the phase. Generating it.')
        game = build_game_from_state(phase["state"])
        possible_orders_original = game.get_all_possible_orders()
        for loc in possible_orders_original:
            phase["possible_orders"][loc].extend(possible_orders[loc])

    # Getting orderable locations
    all_orderable_locations, _ = get_orderable_locs_for_powers(phase["state"], powers)

    # Getting possible orders
    possible_orders = {}
    for loc in all_orderable_locations:
        possible_orders_at_loc = possible_orders_original.get(loc, [])
        if possible_orders_at_loc:
            possible_orders[loc] = possible_orders_at_loc

    # Returning
    return possible_orders


def get_board_alignments(locs, in_adjustment_phase, tokens_per_loc, decoder_length):
    """ Returns a n list of (N_NODES vector) representing the alignments (probs) for the locs on the board state
        :param locs: The list of locs being outputted by the model
        :param in_adjustment_phase: Indicates if we are in A phase (all locs possible at every position) or not.
        :param tokens_per_loc: The number of tokens per loc (TOKENS_PER_ORDER for token_based, 1 for order_based).
        :param decoder_length: The length of the decoder.
        :return: A list of [N_NODES] vector of probabilities (alignments for each location)
    """
    alignments = []

    # Regular phase
    if not in_adjustment_phase:
        for loc in locs:
            alignment = np.zeros([N_NODES], dtype=np.uint8).tolist()
            alignment_index = ALIGNMENTS_INDEX.get(loc[:3], [])
            if loc[:3] not in ALIGNMENTS_INDEX:
                LOGGER.warning('Location %s is not in the alignments index.', loc)
            if alignment_index:
                for index in alignment_index:
                    alignment[index] = 1
            alignments += [alignment] * tokens_per_loc
        if decoder_length != len(locs) * tokens_per_loc:
            LOGGER.warning('Got %d tokens, but decoder length is %d', len(locs) * tokens_per_loc, decoder_length)
        if decoder_length > len(alignments):
            LOGGER.warning('Got %d locs, but the decoder length is %d', len(locs), decoder_length)
            alignments += [np.zeros([N_NODES], dtype=np.uint8).tolist()] * (decoder_length - len(alignments))

    # Adjustment phase (All locs at all positions)
    else:
        alignment = np.zeros([N_NODES], dtype=np.uint8).tolist()
        alignment_index = set()
        for loc in locs:
            if loc[:3] not in ALIGNMENTS_INDEX:
                LOGGER.warning('Location %s is not in the alignments index.', loc)
            for index in ALIGNMENTS_INDEX.get(loc[:3], []):
                alignment_index.add(index)
        if alignment_index:
            for index in alignment_index:
                alignment[index] = 1
        alignments = [alignment] * decoder_length

    # Validating size
    if decoder_length != len(alignments):
        LOGGER.warning('Got %d alignments, but decoder length is %d', len(alignments), decoder_length)

    # Returning
    return np.array(alignments)



    

    


# Vocabulary and constants
PAD_TOKEN = '<PAD>'
GO_TOKEN = '<GO>'
EOS_TOKEN = '<EOS>'
DRAW_TOKEN = '<DRAW>'

def get_order_tokens(order):
    """ Retrieves the order tokens used in an order
        e.g. 'A PAR - MAR' would return ['A PAR', '-', 'MAR']
    """
    # We need to keep 'A', 'F', and '-' in a temporary buffer to concatenate them with the next word
    # We replace 'R' orders with '-'
    # Tokenization would be: 'A PAR S A MAR - BUR' --> 'A PAR', 'S', 'A MAR', '- BUR'
    #                        'A PAR R MAR'         --> 'A PAR', '- MAR'
    buffer, order_tokens = [], []
    for word in order.replace(' R ', ' - ').split():
        buffer += [word]
        if word not in ['A', 'F', '-']:
            order_tokens += [' '.join(buffer)]
            buffer = []
    return order_tokens


def get_vocabulary():
    """ Returns the list of words in the dictionary
        :return: The list of words in the dictionary
    """
    map_object = Map()
    locs = sorted([loc.upper() for loc in map_object.locs])

    vocab = [PAD_TOKEN, GO_TOKEN, EOS_TOKEN, DRAW_TOKEN]                                # Utility tokens
    vocab += ['<%s>' % power_name for power_name in get_map_powers(map_object)]         # Power names
    vocab += ['B', 'C', 'D', 'H', 'S', 'VIA', 'WAIVE']                                  # Order Tokens (excl '-', 'R')
    vocab += ['- %s' % loc for loc in locs]                                             # Locations with '-'
    vocab += ['A %s' % loc for loc in locs if map_object.is_valid_unit('A %s' % loc)]   # Army Units
    vocab += ['F %s' % loc for loc in locs if map_object.is_valid_unit('F %s' % loc)]   # Fleet Units
    return vocab

def get_order_vocabulary():
    """ Computes the list of all valid orders on the standard map
        :return: A sorted list of all valid orders on the standard map
    """
    # pylint: disable=too-many-nested-blocks,too-many-branches
    categories = ['H', 'D', 'B', '-', 'R', 'SH', 'S-',
                  '-1', 'S1', 'C1',                 # Move, Support, Convoy (using 1 fleet)
                  '-2', 'S2', 'C2',                 # Move, Support, Convoy (using 2 fleets)
                  '-3', 'S3', 'C3',                 # Move, Support, Convoy (using 3 fleets)
                  '-4', 'S4', 'C4']                 # Move, Support, Convoy (using 4 fleets)
    orders = {category: set() for category in categories}
    map_object = Map()
    locs = sorted([loc.upper() for loc in map_object.locs])

    # All holds, builds, and disbands orders
    for loc in locs:
        for unit_type in ['A', 'F']:
            if map_object.is_valid_unit('%s %s' % (unit_type, loc)):
                orders['H'].add('%s %s H' % (unit_type, loc))
                orders['D'].add('%s %s D' % (unit_type, loc))

                # Allowing builds in all SCs (even though only homes will likely be used)
                if loc[:3] in map_object.scs:
                    orders['B'].add('%s %s B' % (unit_type, loc))

    # Moves, Retreats, Support Holds
    for unit_loc in locs:
        for dest in [loc.upper() for loc in map_object.abut_list(unit_loc, incl_no_coast=True)]:
            for unit_type in ['A', 'F']:
                if not map_object.is_valid_unit('%s %s' % (unit_type, unit_loc)):
                    continue

                if map_object.abuts(unit_type, unit_loc, '-', dest):
                    orders['-'].add('%s %s - %s' % (unit_type, unit_loc, dest))
                    orders['R'].add('%s %s R %s' % (unit_type, unit_loc, dest))

                # Making sure we can support destination
                if not (map_object.abuts(unit_type, unit_loc, 'S', dest)
                        or map_object.abuts(unit_type, unit_loc, 'S', dest[:3])):
                    continue

                # Support Hold
                for dest_unit_type in ['A', 'F']:
                    for coast in ['', '/NC', '/SC', '/EC', '/WC']:
                        if map_object.is_valid_unit('%s %s%s' % (dest_unit_type, dest, coast)):
                            orders['SH'].add('%s %s S %s %s%s' % (unit_type, unit_loc, dest_unit_type, dest, coast))

    # Convoys, Move Via
    for n_fleets in map_object.convoy_paths:

        # Skipping long-term convoys
        if n_fleets > 4:
            continue

        for start, fleets, dests in map_object.convoy_paths[n_fleets]:
            for end in dests:
                orders['-%d' % n_fleets].add('A %s - %s VIA' % (start, end))
                orders['-%d' % n_fleets].add('A %s - %s VIA' % (end, start))
                for fleet_loc in fleets:
                    orders['C%d' % n_fleets].add('F %s C A %s - %s' % (fleet_loc, start, end))
                    orders['C%d' % n_fleets].add('F %s C A %s - %s' % (fleet_loc, end, start))

    # Support Move (Non-Convoyed)
    for start_loc in locs:
        for dest_loc in [loc.upper() for loc in map_object.abut_list(start_loc, incl_no_coast=True)]:
            for support_loc in (map_object.abut_list(dest_loc, incl_no_coast=True)
                                + map_object.abut_list(dest_loc[:3], incl_no_coast=True)):
                support_loc = support_loc.upper()

                # A unit cannot support itself
                if support_loc[:3] == start_loc[:3]:
                    continue

                # Making sure the src unit can move to dest
                # and the support unit can also support to dest
                for src_unit_type in ['A', 'F']:
                    for support_unit_type in ['A', 'F']:
                        if (map_object.abuts(src_unit_type, start_loc, '-', dest_loc)
                                and map_object.abuts(support_unit_type, support_loc, 'S', dest_loc[:3])
                                and map_object.is_valid_unit('%s %s' % (src_unit_type, start_loc))
                                and map_object.is_valid_unit('%s %s' % (support_unit_type, support_loc))):
                            orders['S-'].add('%s %s S %s %s - %s' %
                                             (support_unit_type, support_loc, src_unit_type, start_loc, dest_loc[:3]))

    # Support Move (Convoyed)
    for n_fleets in map_object.convoy_paths:

        # Skipping long-term convoys
        if n_fleets > 4:
            continue

        for start_loc, fleets, ends in map_object.convoy_paths[n_fleets]:
            for dest_loc in ends:
                for support_loc in map_object.abut_list(dest_loc, incl_no_coast=True):
                    support_loc = support_loc.upper()

                    # A unit cannot support itself
                    if support_loc[:3] == start_loc[:3]:
                        continue

                    # A fleet cannot support if it convoys
                    if support_loc in fleets:
                        continue

                    # Making sure the support unit can also support to dest
                    # And that the support unit is not convoying
                    for support_unit_type in ['A', 'F']:
                        if (map_object.abuts(support_unit_type, support_loc, 'S', dest_loc)
                                and map_object.is_valid_unit('%s %s' % (support_unit_type, support_loc))):
                            orders['S%d' % n_fleets].add(
                                '%s %s S A %s - %s' % (support_unit_type, support_loc, start_loc, dest_loc[:3]))

    # Building the list of final orders
    final_orders = [PAD_TOKEN, GO_TOKEN, EOS_TOKEN, DRAW_TOKEN]
    final_orders += ['<%s>' % power_name for power_name in get_map_powers(map_object)]
    final_orders += ['WAIVE']

    # Sorting each category
    for category in categories:
        category_orders = [order for order in orders[category] if order not in final_orders]
        final_orders += list(sorted(category_orders, key=lambda value: (value.split()[1],        # Sorting by loc
                                                                        value)))                 # Then alphabetically
    return final_orders

def get_power_vocabulary():
    """ Computes a sorted list of powers in the standard map
        :return: A list of the powers
    """
    standard_map = Map()
    return sorted([power_name for power_name in standard_map.powers])


__VOCABULARY__ = get_vocabulary()
VOCABULARY_IX_TO_KEY = {token_ix: token for token_ix, token in enumerate(__VOCABULARY__)}
VOCABULARY_KEY_TO_IX = {token: token_ix for token_ix, token in enumerate(__VOCABULARY__)}
VOCABULARY_SIZE = len(__VOCABULARY__)
del __VOCABULARY__

__ORDER_VOCABULARY__ = get_order_vocabulary()
ORDER_VOCABULARY_IX_TO_KEY = {order_ix: order for order_ix, order in enumerate(__ORDER_VOCABULARY__)}
ORDER_VOCABULARY_KEY_TO_IX = {order: order_ix for order_ix, order in enumerate(__ORDER_VOCABULARY__)}
ORDER_VOCABULARY_SIZE = len(__ORDER_VOCABULARY__)
del __ORDER_VOCABULARY__

__POWER_VOCABULARY__ = get_power_vocabulary()
POWER_VOCABULARY_LIST = __POWER_VOCABULARY__
POWER_VOCABULARY_IX_TO_KEY = {power_ix: power for power_ix, power in enumerate(__POWER_VOCABULARY__)}
POWER_VOCABULARY_KEY_TO_IX = {power: power_ix for power_ix, power in enumerate(__POWER_VOCABULARY__)}
POWER_VOCABULARY_SIZE = len(__POWER_VOCABULARY__)
del __POWER_VOCABULARY__

def token_to_ix(order_token):
    """ Computes the index of an order token in the vocabulary (i.e. order_token ==> token)
        :param order_token: The order token to get the index from (e.g. 'A PAR')
        :return: The index of the order token, a.k.a. the corresponding token (e.g. 10)
    """
    return VOCABULARY_KEY_TO_IX[order_token]

def ix_to_token(token):
    """ Computes the order token at a given index in the vocabulary (i.e. token ==> order_token)
        :param token: The token to convert to an order token (e.g. 10)
        :return: The corresponding order_token (e.g. 'A PAR')
    """
    return VOCABULARY_IX_TO_KEY[max(0, token)]

def order_to_ix(order):
    """ Computes the index of an order in the order vocabulary
        :param order: The order to get the index from
        :return: The index of the order  (None if not found)
    """
    if order in ORDER_VOCABULARY_KEY_TO_IX:
        return ORDER_VOCABULARY_KEY_TO_IX[order]

    # Adjustment for Supporting a move to a coast (stripping the coast)
    words = order.split()
    if len(words) == 7 and words[2] == 'S' and '/' in words[-1]:
        words[-1] = words[-1][:3]
        order = ' '.join([word for word in words])
    return ORDER_VOCABULARY_KEY_TO_IX[order] if order in ORDER_VOCABULARY_KEY_TO_IX else None

def ix_to_order(order_ix):
    """ Computes the order at a given index in the order vocabulary
        :param order_ix: The index of the order to return
        :return: The order at index
    """
    return ORDER_VOCABULARY_IX_TO_KEY[max(0, order_ix)]

PAD_ID = token_to_ix(PAD_TOKEN)
GO_ID = token_to_ix(GO_TOKEN)
EOS_ID = token_to_ix(EOS_TOKEN)
DRAW_ID = token_to_ix(DRAW_TOKEN)

def get_order_based_mask(list_possible_orders, max_length=MAX_CANDIDATES):
    """ Returns a list of candidates ids padded to the max length
        :param list_possible_orders: The list of possible orders (e.g. ['A PAR H', 'A PAR - BUR', ...])
        :return: A list of candidates padded. (e.g. [1, 50, 252, 0, 0, 0, ...])
    """
    candidates = [order_to_ix(order) for order in list_possible_orders]
    candidates = [token for token in candidates if token is not None]
    candidates += [PAD_ID] * (max_length - len(candidates))
    if len(candidates) > max_length:
        LOGGER.warning('Found %d candidates, but only allowing a maximum of %d', len(candidates), max_length)
        candidates = candidates[:max_length]
    return candidates





def get_alignments_index(map_name='standard'):
    """ Computes a list of nodes index for each possible location
        e.g. if the sorted list of locs is ['BRE', 'MAR', 'PAR'] would return {'BRE': [0], 'MAR': [1], 'PAR': [2]}
    """
    current_map = Map(map_name)
    sorted_locs = get_sorted_locs(current_map)
    alignments_index = {}

    # Computing the index of each loc
    for loc in sorted_locs:
        if loc[:3] in alignments_index:
            continue
        alignments_index[loc[:3]] = [index for index, sorted_loc in enumerate(sorted_locs) if loc[:3] == sorted_loc[:3]]
    return alignments_index


# Caching alignments
ALIGNMENTS_INDEX = get_alignments_index()
