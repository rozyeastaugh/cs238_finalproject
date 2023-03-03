import requests
import io
import pandas as pd
import numpy as np
import time


def write_policy(p_dict, filename):
    with open(filename, 'w') as f:
        for a in p_dict.values():
            f.write("{}\n".format(a))


def get_data():
    # TODO: edit this filename if needed/add other files
    raw_url = 'https://raw.githubusercontent.com/rozyeastaugh/cs238_finalproject/main/cs238_finalproject_sarsdata.csv'
    download = requests.get(raw_url).content
    df = pd.read_csv(io.StringIO(download.decode('utf-8')))
    return df


def get_states_actions_rewards(data):
    S = np.array(data['s'])
    A = np.array(data['a'])
    R = np.array(data['r'])
    SP = np.array(data['s'])
    return S, A, R, SP


def simulate(gamma, Q, α, k, data):
    data = np.array(data)
    for i in range(k):
        for d in range(len(data)):
            [s, a, r, sp] = data[d]
            Q[s - 1, a - 1] += α * (r + gamma * max(Q[sp - 1, :]) - Q[s - 1, a - 1])
    return Q


def get_policy(Q, num_states, A, park_limit):
    all_states = np.arange(0, num_states)
    final_policy = {key: None for key in all_states}
    for s in all_states:
        final_policy[s] = max(Q[s])
        '''
        final_policy[s] = A[np.argmax(Q[s])]
        if final_policy[s] == 0:
            final_policy[s] = 0  # NOTE: no need to change action here bc action of 0 is valid
        if final_policy[s] == 1 and (((np.array(list(final_policy.values()))) == 1).sum() - ((np.array(list(final_policy.values()))) == 2).sum()) > 10:
            final_policy[s] = 2  # TODO: check this
        '''
    best_positions = sorted(list(final_policy.keys()), key=lambda i: list(final_policy.values())[i])[-park_limit:]
    for s in all_states:
        if s in best_positions:
            final_policy[s] = 1
        else:
            final_policy[s] = 0
    return final_policy


def compute():
    park_limit = 5
    data = get_data()
    S, A, R, SP = get_states_actions_rewards(data)
    α = 0.01  # learning rate
    gamma = 0.95
    num_states = 100
    num_actions = 3
    Q = np.zeros((num_states, num_actions))
    k = 2
    Q = simulate(gamma, Q, α, k, data)
    final_policy = get_policy(Q, num_states, A, park_limit)
    return final_policy


def q_learning():
    start_time = time.time()

    final_policy = compute()
    outputfilename = 'finalproject_qlearning.policy'
    write_policy(final_policy, outputfilename)

    print("time elapsed: {:.2f}s".format(time.time() - start_time))
    return


q_learning()