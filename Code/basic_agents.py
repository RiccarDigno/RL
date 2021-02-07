import numpy as np
import matplotlib.pyplot as plt

import os, shutil
from tqdm import tqdm

from rl_glue import RLGlue
from environment import BaseEnvironment
from agent import BaseAgent
from optimizer import BaseOptimizer
import plot_script
from randomwalk_environment import RandomWalkEnvironment

from agent import BaseAgent
import tiles3 as tc


def one_hot(state, num_states):
    """
    Given num_state and a state, return the one-hot encoding of the state
    """
    one_hot_vector = np.zeros((1, num_states))
    one_hot_vector[0, int((state - 1))] = 1

    return one_hot_vector


def compute_softmax_prob(actor_w, tiles):
    """
    Computes softmax probability for all actions

    Args:
    actor_w - np.array, an array of actor weights
    tiles - np.array, an array of active tiles

    Returns:
    softmax_prob - np.array, an array of size equal to num. actions, and sums to 1.
    """

    state_action_preferences = []

    for a in range(3):
        state_action_preferences.append(actor_w[a][tiles].sum())

    c = np.max(state_action_preferences)

    numerator = np.exp(state_action_preferences - c)

    denominator = np.sum(numerator)

    softmax_prob = np.divide(numerator, denominator)


    return softmax_prob


def my_matmul(x1, x2):
    """
    Given matrices x1 and x2, return the multiplication of them
    """

    result = np.zeros((x1.shape[0], x2.shape[1]))
    x1_non_zero_indices = x1.nonzero()
    if x1.shape[0] == 1 and len(x1_non_zero_indices[1]) == 1:
        result = x2[x1_non_zero_indices[1], :]
    elif x1.shape[1] == 1 and len(x1_non_zero_indices[0]) == 1:
        result[x1_non_zero_indices[0], :] = x2 * x1[x1_non_zero_indices[0], 0]
    else:
        result = np.matmul(x1, x2)
    return result


def get_value(s, weights):
    """
    Compute value of input s given the weights of a neural network
    """
    # Compute the ouput of the neural network, v, for input s
    psi = my_matmul(s, weights[0]["W"]) + weights[0]["b"]
    x = np.maximum(np.zeros(len(psi)), psi)
    v = my_matmul(x, weights[1]["W"]) + weights[1]["b"]
    return v


def get_gradient(s, weights):
    """
    Given inputs s and weights, return the gradient of v with respect to the weights
    """
    # Compute the gradient of the value function with respect to W0, b0, W1, b1 for input s
    grads = [dict() for i in range(len(weights))]

    psi = my_matmul(s, weights[0]["W"]) + weights[0]["b"]
    x = np.maximum(np.zeros(len(psi)), psi)
    I_x = np.zeros(x.shape)
    I_x[x > 0] = 1
    grads[1]["b"] = np.ones(len(weights[1]["b"]))
    grads[1]["W"] = np.transpose(x)
    grads[0]["b"] = np.multiply(np.transpose(weights[1]["W"]), I_x)
    grads[0]["W"] = my_matmul(np.transpose(s), grads[0]["b"])

    return grads


class SGD(BaseOptimizer):
    def __init__(self):
        pass

    def optimizer_init(self, optimizer_info):
        """Setup for the optimizer.

        Set parameters needed to setup the stochastic gradient descent method.

        Assume optimizer_info dict contains:
        {
            step_size: float
        }
        """
        self.step_size = optimizer_info.get("step_size")

    def update_weights(self, weights, g):
        """
        Given weights and update g, return updated weights
        """
        for i in range(len(weights)):
            for param in weights[i].keys():
                # update weights
                weights[i][param] = weights[i][param] + self.step_size * g[i][param]

        return weights


class Adam(BaseOptimizer):
    def __init__(self):
        pass

    def optimizer_init(self, optimizer_info):
        """Setup for the optimizer.

        Set parameters needed to setup the Adam algorithm.

        Assume optimizer_info dict contains:
        {
            num_states: integer,
            num_hidden_layer: integer,
            num_hidden_units: integer,
            step_size: float,
            self.beta_m: float
            self.beta_v: float
            self.epsilon: float
        }
        """

        self.num_states = optimizer_info.get("num_states")
        self.num_hidden_layer = optimizer_info.get("num_hidden_layer")
        self.num_hidden_units = optimizer_info.get("num_hidden_units")

        # Specify Adam algorithm's hyper parameters
        self.step_size = optimizer_info.get("step_size")
        self.beta_m = optimizer_info.get("beta_m")
        self.beta_v = optimizer_info.get("beta_v")
        self.epsilon = optimizer_info.get("epsilon")

        self.layer_size = np.array([self.num_states, self.num_hidden_units, 1])

        # Initialize Adam algorithm's m and v
        self.m = [dict() for i in range(self.num_hidden_layer + 1)]
        self.v = [dict() for i in range(self.num_hidden_layer + 1)]

        for i in range(self.num_hidden_layer + 1):
            # Initialize self.m[i]["W"], self.m[i]["b"], self.v[i]["W"], self.v[i]["b"] to zero
            self.m[i]["W"] = np.zeros((self.layer_size[i], self.layer_size[i + 1]))
            self.m[i]["b"] = np.zeros((1, self.layer_size[i + 1]))
            self.v[i]["W"] = np.zeros((self.layer_size[i], self.layer_size[i + 1]))
            self.v[i]["b"] = np.zeros((1, self.layer_size[i + 1]))

        # Initialize beta_m_product and beta_v_product to be later used for computing m_hat and v_hat
        self.beta_m_product = self.beta_m
        self.beta_v_product = self.beta_v

    def update_weights(self, weights, g):
        """
        Given weights and update g, return updated weights
        """

        for i in range(len(weights)):
            for param in weights[i].keys():
                # update self.m and self.v
                self.m[i][param] = self.beta_m * self.m[i][param] + (1 - self.beta_m) * g[i][param]
                self.v[i][param] = self.beta_v * self.v[i][param] + (1 - self.beta_v) * (g[i][param] * g[i][param])

                # compute m_hat and v_hat
                m_hat = self.m[i][param] / (1 - self.beta_m_product)
                v_hat = self.v[i][param] / (1 - self.beta_v_product)

                # update weights
                weights[i][param] += self.step_size * m_hat / (np.sqrt(v_hat) + self.epsilon)

        # update self.beta_m_product and self.beta_v_product
        self.beta_m_product *= self.beta_m
        self.beta_v_product *= self.beta_v

        return weights


class TDAgent(BaseAgent):
    def __init__(self):
        self.name = "td_agent"
        pass

    def agent_init(self, agent_info={}):
        """Setup for the agent called when the experiment first starts.

        Set parameters needed to setup the semi-gradient TD with a Neural Network.

        Assume agent_info dict contains:
        {
            num_states: integer,
            num_hidden_layer: integer,
            num_hidden_units: integer,
            step_size: float,
            discount_factor: float,
            self.beta_m: float
            self.beta_v: float
            self.epsilon: float
            seed: int
        }
        """

        # Set random seed for weights initialization for each run
        self.rand_generator = np.random.RandomState(agent_info.get("seed"))

        # Set random seed for policy for each run
        self.policy_rand_generator = np.random.RandomState(agent_info.get("seed"))

        # Set attributes according to agent_info
        self.num_states = agent_info.get("num_states")
        self.num_hidden_layer = agent_info.get("num_hidden_layer")
        self.num_hidden_units = agent_info.get("num_hidden_units")
        self.discount_factor = agent_info.get("discount_factor")

        # Define the neural network's structure
        self.layer_size = np.array([self.num_states, self.num_hidden_layer * self.num_hidden_units, 1])

        # Initialize the neural network's parameter
        self.weights = [dict() for i in range(self.num_hidden_layer + 1)]
        for i in range(self.num_hidden_layer + 1):
            # Initialize self.weights[i]["W"] and self.weights[i]["b"] using self.rand_generator.normal()
            self.weights[i]["W"] = self.rand_generator.normal(0, np.sqrt(2 / self.layer_size[i]),
                                                              (self.layer_size[i], self.layer_size[i + 1]))
            self.weights[i]["b"] = self.rand_generator.normal(0, np.sqrt(2 / self.layer_size[i]),
                                                              (self.layer_size[2], self.layer_size[i + 1]))

        # Specify the optimizer
        self.optimizer = Adam()
        optimizer_info = {"num_states": agent_info["num_states"],
                          "num_hidden_layer": agent_info["num_hidden_layer"],
                          "num_hidden_units": agent_info["num_hidden_units"],
                          "step_size": agent_info["step_size"],
                          "beta_m": agent_info["beta_m"],
                          "beta_v": agent_info["beta_v"],
                          "epsilon": agent_info["epsilon"]}
        self.optimizer.optimizer_init(optimizer_info)

        self.last_state = None
        self.last_action = None

    def agent_policy(self, state):
        # Set chosen_action as 0 or 1 with equal probability.
        chosen_action = self.policy_rand_generator.choice([0, 1])
        return chosen_action

    def agent_start(self, state):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            state (Numpy array): the state from the
                environment's evn_start function.
        Returns:
            The first action the agent takes.
        """
        # select action given state (using self.agent_policy()), and save current state and action (2 lines)
        self.last_state = state
        self.last_action = self.agent_policy(state)

        return self.last_action

    def agent_step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (Numpy array): the state from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            The action the agent is taking.
        """

        # Compute TD error
        current_value = get_value(one_hot(state, self.num_states), self.weights)
        last_value = get_value(one_hot(self.last_state, self.num_states), self.weights)
        target = reward + self.discount_factor * current_value
        delta = target - last_value

        # Retrieve gradients
        grads = get_gradient(one_hot(self.last_state, self.num_states), self.weights)

        # Compute g
        g = [dict() for i in range(self.num_hidden_layer + 1)]
        for i in range(self.num_hidden_layer + 1):
            for param in self.weights[i].keys():
                g[i][param] = delta * grads[i][param]

        # update the weights using self.optimizer
        self.weights = self.optimizer.update_weights(self.weights, g)

        # update self.last_state and self.last_action
        self.last_state = state
        self.last_action = self.agent_policy(state)

        return self.last_action

    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """

        # compute TD error
        last_value = get_value(one_hot(self.last_state, self.num_states), self.weights)
        target = reward
        delta = target - last_value
        # Retrieve gradients
        grads = get_gradient(one_hot(self.last_state, self.num_states), self.weights)

        # Compute g
        g = [dict() for i in range(self.num_hidden_layer + 1)]
        for i in range(self.num_hidden_layer + 1):
            for param in self.weights[i].keys():
                g[i][param] = delta * grads[i][param]

        # update the weights using self.optimizer
        self.weights = self.optimizer.update_weights(self.weights, g)

    def agent_message(self, message):
        if message == 'get state value':
            state_value = np.zeros(self.num_states)
            for state in range(1, self.num_states + 1):
                s = one_hot(state, self.num_states)
                state_value[state - 1] = get_value(s, self.weights)
            return state_value


class DynaQAgent(BaseAgent):

    def agent_init(self, agent_info):
        """Setup for the agent called when the experiment first starts.

        Args:
            agent_init_info (dict), the parameters used to initialize the agent. The dictionary contains:
            {
                num_states (int): The number of states,
                num_actions (int): The number of actions,
                epsilon (float): The parameter for epsilon-greedy exploration,
                step_size (float): The step-size,
                discount (float): The discount factor,
                planning_steps (int): The number of planning steps per environmental interaction

                random_seed (int): the seed for the RNG used in epsilon-greedy
                planning_random_seed (int): the seed for the RNG used in the planner
            }
            :param agent_info:
        """
        try:
            self.num_states = agent_info["num_states"]
            self.num_actions = agent_info["num_actions"]
        except:
            print("You need to pass both 'num_states' and 'num_actions' \
                   in agent_info to initialize the action-value table")
        self.gamma = agent_info.get("discount", 0.95)
        self.step_size = agent_info.get("step_size", 0.1)
        self.epsilon = agent_info.get("epsilon", 0.1)
        self.planning_steps = agent_info.get("planning_steps", 10)

        self.rand_generator = np.random.RandomState(agent_info.get('random_seed', 42))
        self.planning_rand_generator = np.random.RandomState(agent_info.get('planning_random_seed', 42))

        self.q_values = np.zeros((self.num_states, self.num_actions))
        self.actions = list(range(self.num_actions))
        self.past_action = -1
        self.past_state = -1
        self.model = {}

    def update_model(self, past_state, past_action, state, reward):
        """updates the model

        Args:
            past_state       (int): s
            past_action      (int): a
            state            (int): s'
            reward           (int): r
        Returns:
            Nothing
        """
        # Update the model with the (s,a,s',r) tuple
        if past_state not in self.model.keys():
            self.model[past_state] = {past_action: (state, reward)}
        else:
            self.model[past_state].update({past_action: (state, reward)})

    def planning_step(self):
        """performs planning, i.e. indirect RL.

        Args:
            None
        Returns:
            Nothing
        """
        n = 0
        while n < self.planning_steps:
            s = self.planning_rand_generator.choice(list(self.model.keys()))
            a = self.planning_rand_generator.choice(list(self.model[s].keys()))
            (s_p, r) = self.model[s][a]
            if s_p != -1:
                target = r + self.gamma * np.max(self.q_values[s_p, :])
            elif s_p == -1:
                target = r
            self.q_values[s, a] = self.q_values[s, a] + self.step_size * (target - self.q_values[s, a])
            n += 1

    def argmax(self, q_values):
        """argmax with random tie-breaking
        Args:
            q_values (Numpy array): the array of action values
        Returns:
            action (int): an action with the highest value
        """
        top = float("-inf")
        ties = []

        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = []

            if q_values[i] == top:
                ties.append(i)

        return self.rand_generator.choice(ties)

    def choose_action_egreedy(self, state):
        """returns an action using an epsilon-greedy policy w.r.t. the current action-value function.

        Important: assume you have a random number generator 'rand_generator' as a part of the class
                    which you can use as self.rand_generator.choice() or self.rand_generator.rand()

        Args:
            state (List): coordinates of the agent (two elements)
        Returns:
            The action taken w.r.t. the aforementioned epsilon-greedy policy
        """

        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.choice(self.actions)
        else:
            values = self.q_values[state]
            action = self.argmax(values)

        return action

    def agent_start(self, state):
        """The first method called when the experiment starts,
        called after the environment starts.
        Args:
            state (Numpy array): the state from the
                environment's env_start function.
        Returns:
            (int) the first action the agent takes.
        """

        # given the state, select the action using self.choose_action_egreedy()),
        # and save current state and action (~2 lines)
        ### self.past_state = ?
        ### self.past_action = ?

        ### START CODE HERE ###
        self.past_state = state
        self.past_action = self.choose_action_egreedy(state)
        ### END CODE HERE ###

        return self.past_action

    def agent_step(self, reward, state):
        """A step taken by the agent.

        Args:
            reward (float): the reward received for taking the last action taken
            state (Numpy array): the state from the
                environment's step based on where the agent ended up after the
                last step
        Returns:
            (int) The action the agent takes given this state.
        """

        self.q_values[self.past_state, self.past_action] = self.q_values[
                                                               self.past_state, self.past_action] + self.step_size * (
                                                                   reward + self.gamma * np.max(
                                                               self.q_values[state, :]) - self.q_values[
                                                                       self.past_state, self.past_action])
        self.update_model(self.past_state, self.past_action, state, reward)
        self.planning_step()
        self.past_state = state
        self.past_action = self.choose_action_egreedy(state)

        return self.past_action

    def agent_end(self, reward):
        """Called when the agent terminates.

        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """

        self.q_values[self.past_state, self.past_action] = self.q_values[
                                                               self.past_state, self.past_action] + self.step_size * (
                                                                   reward - self.q_values[
                                                               self.past_state, self.past_action])
        self.update_model(self.past_state, self.past_action, -1, reward)
        self.planning_step()


class PendulumTileCoder:
    def __init__(self, iht_size=4096, num_tilings=32, num_tiles=8):
        """
        Initializes the MountainCar Tile Coder
        Initializers:
        iht_size -- int, the size of the index hash table, typically a power of 2
        num_tilings -- int, the number of tilings
        num_tiles -- int, the number of tiles. Here both the width and height of the tiles are the same

        Class Variables:
        self.iht -- tc.IHT, the index hash table that the tile coder will use
        self.num_tilings -- int, the number of tilings the tile coder will use
        self.num_tiles -- int, the number of tiles the tile coder will use
        """

        self.num_tilings = num_tilings
        self.num_tiles = num_tiles
        self.iht = tc.IHT(iht_size)

    def get_tiles(self, angle, ang_vel):
        """
        Takes in an angle and angular velocity from the pendulum environment
        and returns a numpy array of active tiles.

        Arguments:
        angle -- float, the angle of the pendulum between -np.pi and np.pi
        ang_vel -- float, the angular velocity of the agent between -2*np.pi and 2*np.pi

        returns:
        tiles -- np.array, active tiles

        """

        ANGLE_MIN = -np.pi
        ANGLE_MAX = np.pi
        ANG_VEL_MIN = -2 * np.pi
        ANG_VEL_MAX = 2 * np.pi

        angle_scale = self.num_tiles / (ANGLE_MAX - ANGLE_MIN)
        ang_vel_scale = self.num_tiles / (ANG_VEL_MAX - ANG_VEL_MIN)
        ### END CODE HERE ###

        # Get tiles by calling tc.tileswrap method
        # wrapwidths specify which dimension to wrap over and its wrapwidth
        tiles = tc.tileswrap(self.iht, self.num_tilings, [angle * angle_scale, ang_vel * ang_vel_scale],
                             wrapwidths=[self.num_tiles, False])

        return np.array(tiles)


class ActorCriticSoftmaxAgent(BaseAgent):
    def __init__(self):
        self.rand_generator = None

        self.actor_step_size = None
        self.critic_step_size = None
        self.avg_reward_step_size = None

        self.tc = None

        self.avg_reward = None
        self.critic_w = None
        self.actor_w = None

        self.actions = None

        self.softmax_prob = None
        self.prev_tiles = None
        self.last_action = None

    def agent_init(self, agent_info={}):
        """Setup for the agent called when the experiment first starts.

        Set parameters needed to setup the semi-gradient TD(0) state aggregation agent.

        Assume agent_info dict contains:
        {
            "iht_size": int
            "num_tilings": int,
            "num_tiles": int,
            "actor_step_size": float,
            "critic_step_size": float,
            "avg_reward_step_size": float,
            "num_actions": int,
            "seed": int
        }
        """

        # set random seed for each run
        self.rand_generator = np.random.RandomState(agent_info.get("seed"))

        iht_size = agent_info.get("iht_size")
        num_tilings = agent_info.get("num_tilings")
        num_tiles = agent_info.get("num_tiles")

        # initialize self.tc to the tile coder we created
        self.tc = PendulumTileCoder(iht_size=iht_size, num_tilings=num_tilings, num_tiles=num_tiles)

        self.actor_step_size = agent_info.get("actor_step_size") / num_tilings
        self.critic_step_size = agent_info.get("critic_step_size") / num_tilings
        self.avg_reward_step_size = agent_info.get("avg_reward_step_size")

        self.actions = list(range(agent_info.get("num_actions")))

        self.avg_reward = 0.0
        self.actor_w = np.zeros((len(self.actions), iht_size))
        self.critic_w = np.zeros(iht_size)

        self.softmax_prob = None
        self.prev_tiles = None
        self.last_action = None

    def agent_policy(self, active_tiles):
        """ policy of the agent
        Args:
            active_tiles (Numpy array): active tiles returned by tile coder

        Returns:
            The action selected according to the policy
        """

        # compute softmax probability
        softmax_prob = compute_softmax_prob(self.actor_w, active_tiles)

        # Sample action from the softmax probability array
        # self.rand_generator.choice() selects an element from the array with the specified probability
        chosen_action = self.rand_generator.choice(self.actions, p=softmax_prob)

        # save softmax_prob as it will be useful later when updating the Actor
        self.softmax_prob = softmax_prob

        return chosen_action

    def agent_start(self, state):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            state (Numpy array): the state from the environment's env_start function.
        Returns:
            The first action the agent takes.
        """

        angle, ang_vel = state

        active_tiles = self.tc.get_tiles(angle, ang_vel)
        current_action = self.agent_policy(active_tiles)


        self.last_action = current_action
        self.prev_tiles = np.copy(active_tiles)

        return self.last_action

    def agent_step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (Numpy array): the state from the environment's step based on
                                where the agent ended up after the
                                last step.
        Returns:
            The action the agent is taking.
        """

        angle, ang_vel = state

        active_tiles = self.tc.get_tiles(angle, ang_vel)

        delta = reward - self.avg_reward + np.sum(self.critic_w[active_tiles]) - np.sum(self.critic_w[self.prev_tiles])

        self.avg_reward += self.avg_reward_step_size * delta

        self.critic_w[self.prev_tiles] += self.critic_step_size * delta

                self.actor_w[a][self.prev_tiles] += self.actor_step_size * delta * (1 - self.softmax_prob[a])
            else:
                self.actor_w[a][self.prev_tiles] += self.actor_step_size * delta * (0 - self.softmax_prob[a])

        current_action = self.agent_policy(active_tiles)


        self.prev_tiles = active_tiles
        self.last_action = current_action

        return self.last_action

    def agent_message(self, message):
        if message == 'get avg reward':
            return self.avg_reward