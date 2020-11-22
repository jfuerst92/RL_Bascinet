import gym
from gym import envs
import numpy as np
import pandas as pd
import random
import time
import matplotlib.pyplot as plt

def policy_evaluation(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Implement the policy evluation algorithm here given a policy and a complete model of the environment.


    Arguments:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment.
            env.nA is a number of actions in the environment.
        theta: This is the minimum threshold for the error in two consecutive iteration of the value function.
        discount_factor: This is the discount factor - Gamma.

    Returns:
        Vector of length env.nS representing the value function.
    """

    V = np.zeros(env.nS)
    while True:
        delta = 0
        for s in range(env.nS):
            vNew = 0
            for a in range(env.nA):
                for prob, nextState, reward, done in env.P[s][a]:
                    vNew+=policy[s][a] * prob * (reward + discount_factor*V[nextState])

            delta = max(delta, np.abs(V[s]-vNew))
            V[s] = vNew

        if delta < theta:
            break
    return np.array(V)

def policy_iteration(env, policy_eval_fn=policy_evaluation, discount_factor=1.0):
    """
    Implement the Policy Improvement Algorithm here which iteratively evaluates and improves a policy
    until an optimal policy is found.

    Arguments:
        env: The OpenAI envrionment.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.

    Returns:
        A tuple (policy, V).
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.

    """
    start_time = time.time()
    def one_step_lookahead(state, V):
        """
        Implement the function to calculate the value for all actions in a given state.

        Arguments:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.nS

        Returns:
            A vector of length env.nA containing the expected value of each action.
        """
        A = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, nextState, reward, done in env.P[state][a]:
                A[a] += prob * (reward + discount_factor * V[nextState])


        return A

    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA

    numIterations = 0
    mean_vs = []
    times = []
    while True:
        numIterations += 1

        V = policy_eval_fn(policy, env, discount_factor)
        mean_vs.append(np.mean(V))
        #print(np.mean(V))
        policyStable = True

        for s in range(env.nS):
            oldAction = np.argmax(policy[s])

            qValues = one_step_lookahead(s, V)
            newAction = np.argmax(qValues)

            if oldAction != newAction:
                policyStable = False

            policy[s] = np.zeros([env.nA])
            policy[s][newAction] = 1
        times.append(time.time() - start_time)
        if policyStable:
            print(numIterations)
            end_time = time.time() - start_time
            return policy, V, mean_vs, numIterations, times, end_time



    return policy, np.zeros(env.nS), 0, 0, 0, 0

def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    This section is for Value Iteration Algorithm.

    Arguments:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment.
            env.nA is a number of actions in the environment.
        theta: Stop evaluation once value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """
    start_time = time.time()
    def one_step_lookahead(state, V):
        """
        Function to calculate the value for all actions in a given state.

        Arguments:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.nS

        Returns:
            A vector of length env.nA containing the expected value of each action.
        """
        A = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, nextState, reward, done in env.P[state][a]:
                A[a] += prob * (reward + discount_factor * V[nextState])

        return A
    #print(env.nS)
    V = np.zeros(env.nS)

    numIterations = 0
    mean_vs = []
    times = []
    while True:
        numIterations += 1
        delta = 0

        for s in range(env.nS):
            qValues = one_step_lookahead(s, V)
            newValue = np.max(qValues)

            delta = max(delta, np.abs(newValue - V[s]))
            V[s] = newValue

        mean_vs.append(np.mean(V))
        times.append(time.time() - start_time)
        if delta < theta:
            break

    policy = np.zeros([env.nS, env.nA])
    for s in range(env.nS):  #for all states, create deterministic policy
        qValues = one_step_lookahead(s,V)

        newAction = np.argmax(qValues)
        policy[s][newAction] = 1
    end_time = time.time() - start_time
    #print(numIterations)
    return policy, V, mean_vs, numIterations, times, end_time


def same_policy(policyVI, policyPI):
    samePolicy = True
    for s in range(env.nS):
        if samePolicy == False:
            break
        for a in range(env.nA):
            if policyPI[s][a] != policyVI[s][a]:
                samePolicy=False
                break
    if samePolicy:
        print("Policy Iteration and Value Iteration give same Policy")
    else:
        print("Policy Iteration and Value Iteration does not give same Policy")

# Use the following function to see the rendering of the final policy output in the environment
def view_policy(policy):
    curr_state = env.reset()
    counter = 0
    reward = None
    while reward != 20:
        state, reward, done, info = env.step(np.argmax(policy[curr_state]))
        curr_state = state
        counter += 1
        env.s = curr_state
        env.render()

    return counter

"""
polCounter = [view_policy(policyPI) for i in range(1000)]
print(np.average(polCounter))

print('Plot for number of steps for 1000 episodes.')
plt.hist(polCounter)
plt.xlabel('Episode')
plt.ylabel('Number of Steps')
"""
def Q_learning_train(env,alpha,gamma,epsilon,episodes):
    """Q Learning Algorithm with epsilon greedy policy
    Arguments:
        env: Environment
        alpha: Learning Rate --> Extent to which our Q-values are being updated in every iteration.
        gamma: Discount Rate --> How much importance we want to give to future rewards
        epsilon: Probability of selecting random action instead of the 'optimal' action
        episodes: No. of episodes to train
    Returns:
        Q-learning Trained policy
    """

    """Training the agent"""


    #Initialize Q table here
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    epRewards = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()
        done = False
        reward=0
        totalReward = 0

        while not done:
            if random.uniform(0, 1) < epsilon:
                action = random.randint(0, 3)
            else:
                action = np.argmax(q_table[state])

            observation, reward, done, info = env.step(action)
            totalReward += reward

            oldValue = q_table[state, action]
            nextValue = np.max(q_table[observation])

            newValue = (1-alpha) * oldValue + alpha * (reward + gamma * nextValue)

            q_table[state, action] = newValue

            state = observation

        epRewards[i] = totalReward

       # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA

    for state in range(env.nS):
        policy[state] = np.zeros([env.nA])
        newAction = np.argmax(q_table[state])
        policy[state][newAction] = 1



    return policy, q_table, epRewards
