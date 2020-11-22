from hiive import mdptoolbox
import hiive.mdptoolbox.example
import hiive.mdptoolbox.mdp
import hiive.visualization as viz
import time
import matplotlib.pyplot as plt
import numpy as np
import random
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3






def score_frozen_lake(env, policy, episodes=10000, random_pol=False):
    # From https://github.com/realdiganta/solving_openai
    misses = 0
    steps_list = []
    rewards = []
    for episode in range(episodes):
        observation = env.reset()
        steps = 0
        total_reward = 0
        while True:
            if random_pol:
                action = random.randint(0, 3)
            else:
                action = np.argmax(policy[observation])
            observation, reward, done, _ = env.step(action)
            steps += 1
            if done and reward >= 0:
                steps_list.append(steps)
                break
            elif done and reward < 0:
                misses += 1
                break

    mean_number_of_steps = 0 if steps_list == [] else np.mean(steps_list)
    lost_games_perc = (misses / episodes) * 100
    return mean_number_of_steps, lost_games_perc

def print_policy2(policy, map, type, map_size):
    print(policy.shape)
    mapstr = "".join(map)
    if type == "lake":
        #policy_arr = np.asarray(policy)
        dir_arr = []
        idx = 0
        for pol in policy:
            i = np.argmax(pol)
            if mapstr[idx] == 'H':
                dir_arr.append("H")
            elif mapstr[idx] == 'G':
                dir_arr.append("G")
            elif i == 0:
                dir_arr.append("<")
            elif i == 1:
                dir_arr.append("v")
            elif i == 2:
                dir_arr.append(">")
            elif i == 3:
                dir_arr.append("^")


            idx += 1


        #print(policy.shape)
        print()
        res_array = np.array(dir_arr).reshape((map_size, map_size))
        arrow_arr = []
        for row in res_array:
            #arrow_arr.append("".join(row))
            print("".join(row))
        #print(arrow_arr)
        #arrow_str = "".join(map)

def print_policy(policy, type, map_size):
    if type == "lake":
        policy_arr = np.asarray(policy)
        dir_arr = []
        for i in policy_arr:
            if i == 0:
                dir_arr.append("<")
            if i == 1:
                dir_arr.append("v")
            if i == 2:
                dir_arr.append(">")
            if i == 3:
                dir_arr.append("^")


        #print(policy.shape)
        print()
        print(np.array(dir_arr).reshape((map_size, map_size)))

def print_value_fn(V, type, map_size):
    if type == "lake":
        print(np.asarray(V).reshape((map_size, map_size)))

"""
def transform_for_MDPToolbox(env):
    nA, nS = env.nA, env.nS
    P = np.zeros([nA, nS, nS])
    R = np.zeros([nS, nA])
    for s in range(nS):
        for a in range(nA):
            transitions = env.P[s][a]
            for (p_trans, next_s, reward, done) in transitions:
                P[a,s,next_s] += p_trans
                if done and reward == 0.0:
                    reward = -0.75
                R[s,a] = reward
            P[a,s,:] /= np.sum(P[a,s,:])
    # pprint(P)
    return P, R

def create_matrices(env, state_num, action_num):
    P = np.zeros((action_num, state_num, state_num))
    R = np.zeros((state_num, action_num))
    for state in range(state_num):
        for action in range(action_num):
            #print(state)
            #print(action)
            #print("---")
            for prob_tuple in env.env.P[state][action]:
                probability, state_prime, reward, terminated = prob_tuple
                P[action, state, state_prime] = probability
                #print(reward)
                R[state, action] = reward
                #print(P)
                P[action, state, :] = P[action, state, :] / np.sum(P[action, state, :])
                #print(P)f
    return(P, R)
"""
def plot_result(result, name):
    reward = []
    error = []
    time = []
    max_v = []
    mean_v = []
    iter = []
    #state = []
    for it in result:
        reward.append(it['Reward'])
        error.append(it['Error'])
        time.append(it['Time'])
        max_v.append(it['Max V'])
        mean_v.append(it['Mean V'])
        iter.append(it['Iteration'])
        #state.append(it['State'])

    fig, axs = plt.subplots(2, 2)
    fig.suptitle(name)
    axs[0, 0].plot(iter, reward, label="reward")
    axs[0, 0].set_title("reward per iteration")
    axs[0, 0].set(xlabel='iteration', ylabel='reward')
    axs[0, 1].plot(iter, error, label="error")
    axs[0, 1].set_title("error per iteration")
    axs[0, 1].set(xlabel='iteration', ylabel='error')
    axs[1, 0].plot(iter, time, label="time")
    axs[1, 0].set_title("time per iteration")
    axs[1, 0].set(xlabel='iteration', ylabel='time')
    axs[1, 1].plot(iter, mean_v, label="mean V")
    axs[1, 1].set_title("mean V per iteration")
    axs[1, 1].set(xlabel='iteration', ylabel='mean V')
    plt.show()
    plt.clf()



def run_with_n_states(P, R, mdp, n_states_arr=[4, 8, 16, 32, 64, 128]):
    iters = []
    times = []
    policies = []
    for n_states in n_states_arr:
        #P, R = mdptoolbox.example.forest(S=n_states, r1=4, r2=2)
        mdp_inst = mdp(P, R, 0.9)
        result = mdp_inst.run()
        times.append(mdp_inst.time)
        iters.append(mdp_inst.iter)
        policies.append(mdp_inst.policy)
    return iters, times, policies

def run_with_discounts(P, R, mdp, discount_arr=np.arange(0.01, 0.99, 0.01)):
    iters = []
    times = []
    policies = []
    rewards = []
    for n_states in n_states_arr:
        #P, R = mdptoolbox.example.forest(S=n_states, r1=4, r2=2)
        mdp_inst = mdp(P, R, 0.9)
        result = mdp_inst.run()
        times.append(mdp_inst.time)
        iters.append(mdp_inst.iter)
        policies.append(mdp_inst.policy)
        reward.append(mdp_inst.reward)
    return iters, times, policies

def run_experiment(P, R, visualize=True):
    n_states_arr=range(3, 20)
    vi_iters, vi_times, vi_policies = run_with_n_states(P, R,
                            hiive.mdptoolbox.mdp.ValueIteration,
                            n_states_arr
                        )
    pi_iters, pi_times, pi_policies = run_with_n_states(P, R,
                            hiive.mdptoolbox.mdp.PolicyIteration,
                            n_states_arr
                        )
    if visualize:
        plt.plot(n_states_arr, pi_times, label="PI times")
        plt.plot(n_states_arr, vi_times, label="VI times")
        title = 'times per problem size'
        plt.title(title)
        plt.xlabel('# states')
        plt.ylabel('time(s)')
        plt.grid()
        plt.legend()
        plt.show()
        plt.clf()

        plt.plot(n_states_arr, pi_iters, label="PI iters")
        plt.plot(n_states_arr, vi_iters, label="VI iters")
        title = 'iters per problem size'
        plt.title(title)
        plt.xlabel('# states')
        plt.ylabel('# iters')
        plt.grid()
        plt.legend()
        plt.show()
        plt.clf()

def run_q_learning(P, R, map_size, visualize=True):
    #P, R = mdptoolbox.example.forest(S=2000, r1=4, r2=2)
    ql = mdptoolbox.mdp.QLearning(P, R, gamma=0.95, alpha=0.01,
                   alpha_decay=0.99, alpha_min=0.001,
                   epsilon=1.0, epsilon_min=0.1,
                   epsilon_decay=0.99,
                    n_iter=10000)
    result = ql.run()
    if visualize:
        plot_result(result, "QLearning")
        print("Q time:", ql.time)
        print(len(result))
        plt.plot(len(ql.mean_discrepancy), ql.mean_discrepancy, label="VI iters")
        title = 'iters per problem size'
        plt.title(title)
        plt.xlabel('# states')
        plt.ylabel('# iters')
        plt.grid()
        plt.legend()
        plt.show()
        plt.clf()
    print_policy(ql.policy, "lake", map_size)

    return ql.policy

def run_pi(P, R, map_size, visualize=True):
    #P, R = mdptoolbox.example.forest(S=2000, r1=4, r2=2)
    pi = mdptoolbox.mdp.PolicyIteration(P, R, 0.9)
    result = pi.run()
    print_policy(pi.policy, "lake", map_size)
    #print(pi.V)
    print_value_fn(pi.V, "lake", map_size)
    return pi.policy

def run_vi(P, R, map_size, visualize=True):
    #P, R = mdptoolbox.example.forest(S=2000, r1=4, r2=2)
    vi = mdptoolbox.mdp.ValueIteration(P, R, 0.9)
    result = vi.run()
    print_policy(vi.policy, "lake", map_size)
    print_value_fn(vi.V, "lake", map_size)
    return vi.policy

def tune_ql(P, R, map_size, arg_name, arg_range):
    #P, R = mdptoolbox.example.forest(S=2000, r1=4, r2=2)

    args = {
        'transitions': P,
        'reward': R,
        'gamma': 0.95,
        'alpha': 0.1,
        'alpha_decay': 0.99,
        'alpha_min': 0.001,
        'epsilon': 1.0,
        'epsilon_min': 0.1,
        'epsilon_decay': 0.99,
        'n_iter':10000
    }
    times = []
    #iters = []
    for arg_val in arg_range:
        args[arg_name] = arg_val
        ql = mdptoolbox.mdp.QLearning(**args)
        result = ql.run()
        times.append(ql.time)
        #iters.append(ql.iter)

    plt.plot(arg_range, times, label="QL times")
    title = 'times vs ' + arg_name
    plt.title(title)
    plt.xlabel(arg_name)
    plt.ylabel('time(s)')
    plt.grid()
    plt.legend()
    plt.show()
    plt.clf()




def analyze_iteration_results(P, R):
    #P, R = mdptoolbox.example.forest(S=2000, r1=4, r2=2)
    vi = mdptoolbox.mdp.ValueIteration(P, R, 0.9)
    result = vi.run()
    plot_result(result, "Value Iteration")
    pi = mdptoolbox.mdp.PolicyIteration(P, R, 0.9)
    result = pi.run()
    plot_result(result, "Policy Iteration")
    ql = mdptoolbox.mdp.QLearning(P, R, 0.9)
    ql.run()
    if pi.policy == vi.policy:
        print("both are equal")

#analyze_iteration_results()
#run_experiment()
#run_q_learning()
#tune_ql('alpha_decay', np.arange(0.1, 0.9, 0.1))
