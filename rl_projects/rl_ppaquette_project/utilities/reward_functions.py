
import logging
from enum import Enum
from abc import ABCMeta, abstractmethod

from diplomacy import Game, Map

from utilities.utility_functions import ALL_STANDARD_POWERS



# --- Defaults ---
DEFAULT_PENALTY = 0.
DEFAULT_GAMMA = 0.99

LOGGER = logging.getLogger(__name__)

class DoneReason(Enum):
    """ Enumeration of reasons why the environment has terminated """
    NOT_DONE = 'not_done'                                       # Partial game - Not yet completed
    GAME_ENGINE = 'game_engine'                                 # Triggered by the game engine
    AUTO_DRAW = 'auto_draw'                                     # Game was automatically drawn accord. to some prob.
    PHASE_LIMIT = 'phase_limit'                                 # The maximum number of phases was reached
    THRASHED = 'thrashed'     

class AbstractRewardFunction(metaclass=ABCMeta):
    """ Abstract class representing a reward function """

    @property
    @abstractmethod
    def name(self):
        """ Returns a unique name for the reward function """
        raise NotImplementedError()

    @abstractmethod
    def get_reward(self, previous_state, state, power_name, is_terminal_state, done_reason):
        """ Computes the reward for a given power
            :param previous_state: The `.proto.State` representation of the last state of the game (before .process)
            :param state: The `.proto.State` representation of the state of the game (after .process)
            :param power_name: The name of the power for which to calculate the reward (e.g. 'FRANCE').
            :param is_terminal_state: Boolean flag to indicate we are at a terminal state.
            :param done_reason: An instance of DoneReason indicating why the terminal state was reached.
            :return: The current reward (float) for power_name.
            :type done_reason: diplomacy_research.models.gym.environment.DoneReason | None
        """
        raise NotImplementedError()

    def get_episode_rewards(self, saved_game, power_name):
        """ Compute the list of all rewards for a saved game.
            :param saved_game_proto: A `.proto.SavedGame` representation
            :param power_name: The name of the power for which we want to compute rewards
            :return: A list of rewards (one for each transition in the saved game)
        """
        # Restoring cached rewards
        # if saved_game_proto.reward_fn == self.name and saved_game_proto.rewards[power_name].value:
        #     return list(saved_game_proto.rewards[power_name].value)

        # Otherwise, computing them
        episode_rewards = []
        done_reason = DoneReason(saved_game["done_reason"]) if saved_game["done_reason"] != '' else None

        # Making sure we have at least 2 phases (1 transition)
        n_phases = len(saved_game["phases"])
        if n_phases < 2:
            return episode_rewards

        # Computing the reward of each phase (transition)
        for phase_index in range(n_phases - 1):
            current_state = saved_game["phases"][phase_index]["state"]
            next_state = saved_game["phases"][phase_index + 1]["state"]
            is_terminal = phase_index == n_phases - 2
            if is_terminal:
                episode_rewards += [self.get_reward(current_state,
                                                    next_state,
                                                    power_name,
                                                    is_terminal_state=True,
                                                    done_reason=done_reason)]
            else:
                episode_rewards += [self.get_reward(current_state,
                                                    next_state,
                                                    power_name,
                                                    is_terminal_state=False,
                                                    done_reason=None)]
        return episode_rewards



class CustomIntUnitReward(AbstractRewardFunction):
    """ Reward function: greedy supply center.
        This reward function attempts to maximize the number of supply centers in control of a power.
        The reward is the gain/loss of supply centers between the previous and current phase (+1 / -1)
        The reward is given as soon as a SC is touched (rather than just during the Adjustment phase)
        Homes are also +1/-1.
        The reward is given at every phase (i.e. it is an intermediary reward).
    """
    @property
    def name(self):
        """ Returns a unique name for the reward function """
        return 'custom_int_unit_reward'

    def get_reward(self, previous_state, state, power_name, is_terminal_state, done_reason):
        """ Computes the reward for a given power
            :param previous_state: The `.proto.State` representation of the last state of the game (before .process)
            :param state: The `.proto.State` representation of the state of the game (after .process)
            :param power_name: The name of the power for which to calculate the reward (e.g. 'FRANCE').
            :param is_terminal_state: Boolean flag to indicate we are at a terminal state.
            :param done_reason: An instance of DoneReason indicating why the terminal state was reached.
            :return: The current reward (float) for power_name.
            :type done_reason: diplomacy_research.models.gym.environment.DoneReason | None
        """
        assert done_reason is None or isinstance(done_reason, DoneReason), 'done_reason must be a DoneReason object.'
        if power_name not in state["centers"].keys() or power_name not in previous_state["centers"].keys():
            if power_name not in ALL_STANDARD_POWERS:
                LOGGER.error('Unknown power %s. Expected powers are: %s', power_name, ALL_STANDARD_POWERS)
            return 0.

        map_object = Map(state["map"])
        n_centers_req_for_win = len(map_object.scs) // 2 + 1.
        current_centers = set(state["centers"][power_name])
        prev_centers = set(previous_state["centers"][power_name])
        all_supply_centers = map_object.scs

        if done_reason == DoneReason.THRASHED and current_centers:
            return -1. * n_centers_req_for_win

        # Adjusting supply centers for the current phase
        # Dislodged units don't count for adjustment
        for unit_power in state["units"].keys():
            if unit_power == power_name:
                for unit in state["units"][unit_power]:
                    if '*' in unit:
                        continue
                    unit_location = unit[2:5]
                    if unit_location in all_supply_centers and unit_location not in current_centers:
                        current_centers.add(unit_location)
            else:
                for unit in state["units"][unit_power]:
                    if '*' in unit:
                        continue
                    unit_location = unit[2:5]
                    if unit_location in all_supply_centers and unit_location in current_centers:
                        current_centers.remove(unit_location)

        # Adjusting supply centers for the previous phase
        # Dislodged units don't count for adjustment
        for unit_power in previous_state["units"].keys():
            if unit_power == power_name:
                for unit in previous_state["units"][unit_power]:
                    if '*' in unit:
                        continue
                    unit_location = unit[2:5]
                    if unit_location in all_supply_centers and unit_location not in prev_centers:
                        prev_centers.add(unit_location)
            else:
                for unit in previous_state["units"][unit_power]:
                    if '*' in unit:
                        continue
                    unit_location = unit[2:5]
                    if unit_location in all_supply_centers and unit_location in prev_centers:
                        prev_centers.remove(unit_location)

        # Computing difference
        gained_centers = current_centers - prev_centers
        lost_centers = prev_centers - current_centers

        # Computing reward
        return float(len(gained_centers) - len(lost_centers))


class ProportionalReward(AbstractRewardFunction):
    """ Proportional scoring system.
        For win - The winner takes all, the other parties get nothing
        For draw - The pot size is divided by the number of n of sc^i / sum of n of sc^i.
        All non-terminal states receive 0.
    """
    def __init__(self, pot_size=34, exponent=1):
        """ Constructor
            :param pot_size: The number of points to split across survivors (default to 34, which is the nb of centers)
            :param exponent: The exponent to use in the draw calculation.
        """
        assert pot_size > 0, 'The size of the pot must be positive.'
        self.pot_size = pot_size
        self.i = exponent

    @property
    def name(self):
        """ Returns a unique name for the reward function """
        return 'proportional_reward'

    def get_reward(self, previous_state, state, power_name, is_terminal_state, done_reason):
        """ Computes the reward for a given power
            :param previous_state: The `.proto.State` representation of the last state of the game (before .process)
            :param state: The `.proto.State` representation of the state of the game (after .process)
            :param power_name: The name of the power for which to calculate the reward (e.g. 'FRANCE').
            :param is_terminal_state: Boolean flag to indicate we are at a terminal state.
            :param done_reason: An instance of DoneReason indicating why the terminal state was reached.
            :return: The current reward (float) for power_name.
            :type done_reason: diplomacy_research.models.gym.environment.DoneReason | None
        """
        assert done_reason is None or isinstance(done_reason, DoneReason), 'done_reason must be a DoneReason object.'
        if power_name not in state["centers"]:
            if power_name not in ALL_STANDARD_POWERS:
                LOGGER.error('Unknown power %s. Expected powers are: %s', power_name, ALL_STANDARD_POWERS)
            return 0.
        if not is_terminal_state:
            return 0.
        if done_reason == DoneReason.THRASHED:
            return 0.

        map_object = Map(state["map"])
        n_centers_req_for_win = len(map_object.scs) // 2 + 1.
        victors = [power for power in state["centers"].keys()
                   if len(state["centers"][power]) >= n_centers_req_for_win]

        # if there is a victor, winner takes all
        # else survivors get points according to nb_sc^i / sum of nb_sc^i
        if victors:
            split_factor = 1. if power_name in victors else 0.
        else:
            denom = {power: len(state["centers"][power]) ** self.i for power in state["centers"].keys()}
            split_factor = denom[power_name] / float(sum(denom.values()))
        return self.pot_size * split_factor

    

class MixProportionalCustomIntUnitReward(AbstractRewardFunction):
    """ 50% of ProportionalReward
        50% of CustomIntUnitReward
    """
    def __init__(self):
        """ Constructor """
        self.proportional = ProportionalReward()
        self.custom_int_unit = CustomIntUnitReward()

    @property
    def name(self):
        """ Returns a unique name for the reward function """
        return 'mix_proportional_custom_int_unit'

    def get_reward(self, previous_state, state, power_name, is_terminal_state, done_reason):
        """ Computes the reward for a given power
            :param previous_state: The `.proto.State` representation of the last state of the game (before .process)
            :param state: The `.proto.State` representation of the state of the game (after .process)
            :param power_name: The name of the power for which to calculate the reward (e.g. 'FRANCE').
            :param is_terminal_state: Boolean flag to indicate we are at a terminal state.
            :param done_reason: An instance of DoneReason indicating why the terminal state was reached.
            :return: The current reward (float) for power_name.
            :type done_reason: diplomacy_research.models.gym.environment.DoneReason | None
        """
        assert done_reason is None or isinstance(done_reason, DoneReason), 'done_reason must be a DoneReason object.'
        return 0.5 * self.proportional.get_reward(previous_state,
                                                  state,
                                                  power_name,
                                                  is_terminal_state,
                                                  done_reason) \
               + 0.5 * self.custom_int_unit.get_reward(previous_state,
                                                       state,
                                                       power_name,
                                                       is_terminal_state,
                                                       done_reason)

class DefaultRewardFunction(MixProportionalCustomIntUnitReward):
    """ Default reward function class """
    # source code shows nothing else here :9
