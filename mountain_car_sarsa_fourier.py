import gym, itertools
import numpy as np
import matplotlib.pyplot as plt
import sys

# Initializations
num_episodes = 3000  # 1000
num_timesteps = 800  # 200
# Don't need to understand
gym.envs.register(
    id="MountainCarLongerEpisodeLength-v0",
    entry_point="gym.envs.classic_control:MountainCarEnv",
    max_episode_steps=num_timesteps,  # MountainCar-v0 uses 200
    reward_threshold=-110.0,
)
env = gym.make("MountainCarLongerEpisodeLength-v0")
# Represents the number of possible actions
num_actions = env.action_space.n
dim = env.observation_space.high.size

# Parameters
# Order of the basis functions
order = 5
# Total number of individual basis vectors
num_ind = int(pow(order + 1.0, dim))
# Multipliers are the coefficient vectors used within the computation of cosine or sine function computation
multipliers = np.zeros((num_ind, dim))

# Hyperparameters that can be used to optimize our learning algorithm
epsilon = 0.1
Lambda = 0.5
alpha = 0.00005
gamma = 0.99


# Used to normalize the state by demeaning, don't need to understand
xbar = np.zeros((2, dim))
xbar[0, :] = env.observation_space.low
xbar[1, :] = env.observation_space.high

ep_length = np.zeros(num_episodes)
np.set_printoptions(precision=2)
# These are the weights which the basis functions are multiplied to get a value
weights = np.zeros((num_ind, num_actions))
# This function returns a normalized state where all variable values are between 0 and 1.
def normalize_state(s):
    y = np.zeros(len(s))
    for i in range(len(s)):
        if s[i] > xbar[1, i]:
            y[i] = 1
        elif s[i] < xbar[0, i]:
            y[i] = 0
        else:
            y[i] = (s[i] - xbar[0, i]) / (xbar[1, i] - xbar[0, i])
    return y


# TODO
# Returns an ndarray basis functions
def phi(state):
    "*** Fill in code to return the computed basis functions! ***"
    new_basis = np.matmul(multipliers, state)
    new_basis = new_basis * np.pi
    return np.cos(new_basis)


# TODO
# Create the fourier basis coefficients
def create_multipliers():
    global multipliers
    "*** Fill in the code to generate the coefficients for fourier " "basis functions and assign it to the variable multipliers***"
    col_index = 0
    for i in range(order + 1):
        for j in range(order + 1):
            multipliers[col_index][0] = i
            multipliers[col_index][1] = j
            col_index += 1


# Returns the value of an action at some state
def action_value(input_state, action):
    state_normalized = normalize_state(input_state)
    features = phi(state_normalized)
    Qvalue = np.dot(weights[:, action], features)
    return Qvalue


# Returns an exploratory action for an input
def learning_policy(input_state):
    global weights
    # Normalizing states
    state_normalized = normalize_state(input_state)
    # Converting state space to features
    features = phi(state_normalized)
    # Using weights array to find the possible actions
    Qvalues = np.dot(features, weights)
    random_num = np.random.random()
    # Selecting the best action
    if random_num < 1.0 - epsilon:
        action_chosen = Qvalues.argmax()
    else:
        action_chosen = env.action_space.sample()
    # Returning the best action (represented as an integer)
    return int(action_chosen)


# WILL NOT BE USED
def test_policy(state):
    global weights
    # Normalizing the states
    state_normalized = normalize_state(state)
    # Converting to higher dimensional space using the basis functions
    features = phi(state_normalized)
    # Multiplying by learned weights
    Qvalues = np.dot(features, weights)
    # Returning the best policy
    return int(Qvalues.argmax())


# TODO
def SARSA_Learning():
    """
    Implement Sarsa-lambda algorithm to update qtable, etable and learning policy.
    Input:
        all parameters
    Output:
        This function does not return an output. It only updates the weights according to the algorithm.
    """
    global weights
    for ep in range(num_episodes):
        e = np.zeros((num_ind, num_actions))
        state = env.reset()
        norm_state = normalize_state(state)
        features = phi(norm_state)
        action = learning_policy(state)
        rewards = np.array([0])
        # Each episode
        for t in range(num_timesteps):
            next_state, reward, done, info = env.step(action)
            rewards = np.append(rewards, reward)
            "*** Fill in the rest of the algorithm!***"
            # Updating the eligibility trace.
            e = gamma * Lambda * e
            e[:, action] += phi(normalize_state(state))
            # Setting the value of q, depending on if we are in an absorbing state.
            if done:
                new_q = 0
            else:
                new_action = learning_policy(next_state)
                new_q = action_value(next_state, new_action)
            # Updating the delta value with current and old values.
            delta = reward + gamma * new_q - action_value(state, action)
            # Updating the weight values
            weights = weights + alpha * delta * e
            if done:
                break
            state = next_state
            action = new_action

        # Visualization code
        ep_length[ep] = t
        print(np.sum(rewards))

    # Saving our weights to the NumpY]]]]
    np.save("mountain_car_saved_weights_grading.npy", weights)


def SARSA_Test(num_test_episodes):
    for ep in range(num_test_episodes):
        state = env.reset()
        rewards = np.array([0])

        # Each episode
        for t in range(num_timesteps):
            # env.render()
            action = test_policy(state)
            next_state, reward, done, info = env.step(action)
            state = next_state
            rewards = np.append(rewards, reward)
            if done:
                print((rewards.shape))
                break

        print(np.sum(rewards))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        "*** run without input arguments to learn ***"
        create_multipliers()
        SARSA_Learning()
    elif sys.argv[1] == "test":
        "*** run to test the saved policy with input argument test ***"
        create_multipliers()
        weights = np.load("mountain_car_saved_weights_grading.npy")
        num_test_episodes = 100
        SARSA_Test(num_test_episodes)
    else:
        print("unknown input argument")
