import sys
import requests
import io
import pandas as pd
import random
import math
import numpy as np
import time


def get_gaussian_probabilities(size, mean, cov):
    prob_dist = np.random.multivariate_normal(mean, cov, size=size)
    return prob_dist


def lookahead(U, s, a, S, T, R, gamma):
    return R(s, a) + gamma * sum(T(s, a, s_prime)*U[s_prime] for s_prime in S)  # TODO: check this


def forward_search(s, d, U, A, S, T, R, gamma):
    if d <= 0:
        return (None, U(s))  # (a, u)
    best = (None, math.inf)  # (a, u)
    U_prime = forward_search(s, d-1, U)[1]  # recurse, get second index which is u
    for a in A:
        u = lookahead(U_prime, s, a, S, T, R, gamma)  # (a, u)
        if u > best[1]:  # best[1] = u
            best = (a, u)
    return best  # (a, u)


def simulate(prob_dist, S):
    F = []
    for s in range(len(S)):
        if random.random() < prob_dist[s]:  # TODO: check if need to convert to linear index, check if < is right
            F.append(True)
        else:
            F.append(False)
    return F


def recursive_fwd_search(best_score, prob_dist, S):
    for t in range(100):  # run for 100 seconds/timesteps
        pass
        # TODO: On each timestep, choose one position to place a park
        position = random.randint(0, 100)
        while S[position][1] == 1:  # if already a park there, choose another one
            position = random.randint(0, 100)
        S[position][1] = 1  # place park there, change second index of tuple to 1

        # TODO: Expose to flooding, how to simulate given prob_dist? output booleans (yes flooded or no didnt flood) for each cell
        F = simulate(prob_dist, S)

        # TODO: Recurse to next subtree, keeping track of best score overall
        # TODO****: make it like k2 graph search, where you add a park then calculate score then remove it? implement fwd search like in book?!

        # TODO: Return best tree path (i.e. optimal placement of 5 parks)


def fwd_search():
    start_time = time.time()

    # Set size of problem
    size = (10, 10)

    # Set limit of number of parks we can add given current budget
    park_limit = 5
    d = park_limit  # TODO: ask, this will also be the depth of our search, correct?

    # Define probability distribution for flooding in each of the grid ares
    mean = [5, 5]
    cov = [[5, 0], [5, 0]]
    prob_dist = get_gaussian_probabilities(size, mean, cov)

    # Formulate MDP
    # TODO: should a be the index of whatever park we're adding? so when we fwd search the whole tree we get all combos of the 5 parks and find which combo is best?
    A = [0, 1, 2]  # 0: do nothing; 1: add park; 2: remove park
    S = []
    num_states = 100
    for i in range(num_states):
        # Initialize states as (position, whether or not it has a park there, probability of flooding)
        S.append((i, 0, prob_dist[i]))  # TODO: check if this indexing of prob_dist is right - have to change to linear index?
    # R = ? # TODO: how to find this / what to initialize it as?
    # T = ? # TODO: how to find this / what to initialize it as?
    # U = ????????  # TODO: how to find this / what to initialize it as?
    gamma = 0.95  # TODO: is this right?

    # Find optimal placement of parks using forward search
    # TODO: need to get policy first in order to get d and U??
    # TODO: need to iterate through s to calculate this?
    for s in S:
        policy = forward_search(s, d, U, S, T, R, gamma)[0]

    # --------------------------------------------------------------
    # TODO: OR SCRATCH ^all that, INSTEAD DO BELOW:
    best_score = -math.inf
    recursive_fwd_search(best_score, prob_dist, S)

    #final_policy = compute(inputfilename)
    #write_policy(final_policy, outputfilename)

    print("time elapsed: {:.2f}s".format(time.time() - start_time))
    return

if __name__ == '__fwd_search__':
    fwd_search()