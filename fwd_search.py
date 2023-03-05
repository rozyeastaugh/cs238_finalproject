import sys
import requests
import io
import pandas as pd
import random
import math
import numpy as np
import time
from scipy.stats import multivariate_normal


def write_policy(policy, filename):
    with open(filename, 'w') as f:
        for a in policy:  # TODO: check if policy is list, change if not
            f.write("{}\n".format(a))


def random_probs(num_states):
    probs = np.zeros(num_states)
    for s in range(num_states):
        probs[s] = random.random()
    return probs


def uniform_probs(num_states):
    return [.5]*num_states


def gaussian_probs(num_states):
    mean = [5, 5]
    cov = [[5, 0], [0, 5]]
    dist = multivariate_normal(mean, cov)
    weights = []
    for i in range(10):  # Calculate pdf for all grid points
        for j in range(10):
            weights.append(dist.pdf([i, j]))
    probs = weights/sum(weights)
    return probs


def get_probabilities(num_states):
    return gaussian_probs(num_states)  # TODO: try uniform and random as well for final paper


def transition(T, prob_dist, a, s, sp):
    T[s, a, sp] = (1 if sp == a else 0, 0 if sp == a else random.random() < prob_dist[s])  # 3D 100x100x100 matrix
    return T  # TODO: make sure T correctly updated


def lookahead(U, s, a, S, T, R, gamma, prob_dist):  # TODO: bad to add prob_dist here?
    # NOTE: Marc's other idea for keeping track of all 5 actions: also return actual U values here... ???
    # TODO: change to s not S? i.e. below code not current code
    '''
    u_new = []
    for s in S:
        u_new.append(R[s, a] + gamma * sum(transition(T, prob_dist, a, s, sp)*U[sp] for sp in S))
    return u_new, T
    '''
    return R[s, a] + gamma * sum(transition(T, prob_dist, a, s, sp)*U[sp] for sp in S), T


def simulate(prob_dist, a, S, R):
    for s in range(len(S)):
        if random.random() < prob_dist[s]:
            s[1] = 1
            R[s] = -10  # If flooded, reward negatively TODO: try diff values here
        else:
            s[1] = 0
            R[s] = 10  # # If not flooded, reward poistively TODO: try diff values here
    score = sum(R)
    return score, R, S


def forward_search(S, d, U, A, T, R, gamma, prob_dist, best_score, best_comb_parks):
    if d <= 0:
        return (None, U(s))  # (a, u)  # TODO: figure out how to find U(s) given that we're looking at at all S
    best = (None, math.inf)  # (a, u)
    U_prime = forward_search(S, d, U, A, T, R, gamma, prob_dist, best_score, best_comb_parks)[1][1]
    # OLD: U_prime = forward_search(s, d - 1, U)[1]  # recurse, get second index which is u
    for a in A:
        # Simulate Flooding
        S[a][0] = 1  # Place park there, change second index of tuple to 1
        prob_dist[a] = 0  # Change prob of flooding to zero (TODO: need to input last component of all S's in simulate below??)
        score, R, S = simulate(prob_dist, a, S, R)  # TODO: make sure R and S are correctly updated

        # Get next step of value function
        u, T = lookahead(U_prime, s, a, S, T, R, gamma)  # (a, u)  # TODO: dont input s, just S

        # Save if best  # TODO: this will only save 1 action, not 5, right?!?
        if u > best[1]:  # best[1] = u
            best = (a, u)
        if score > best_score:  # NOTE: Write about this in project: experimenting with different metrics; added this cause unsure if ranking by utility is best or if it's better to rank by my more specific score metric; Marc said try both and write about it!
            best_score = score
            best = (a, score)
            best_comb_parks.append(a)
    best_comb_parks.append(best[0]) # TODO: Check that this correctly adds previously encoded action to current best action in order to return best five actions
    return best, best_score, best_comb_parks


def fwd_search():
    start_time = time.time()

    # Set size of problem
    num_states = 10*10

    # Set limit of number of parks we can add given current budget
    park_limit = 5
    d = park_limit

    # Define probability distribution for flooding in each of the grid ares
    prob_dist = get_probabilities(num_states)

    # Formulate/Initialize MDP
    S = []
    A = np.arange(0, num_states)
    for i in range(num_states): # Initialize states as (whether it has a park there, whether it flooded or not) # NOTE: removed position bc encoded as index of s in S, took out prob of flood bc need deterministic states
        S.append((0, random.random() < prob_dist[i]))
    R = np.zeros(num_states)
    T = np.zeros([num_states, num_states, num_states])
    U = np.zeros(num_states)
    gamma = 0.95  # TODO: play around with diff values

    # Find optimal placement of parks using forward search
    best_score = -math.inf
    best_comb_parks = []
    # TODO: need to run fwd search in loop like this, or initialize random s and run fwd search once given any initial state?

    '''
    # NOTE: Marc's other idea for getting all 5 actions was to run fwd search 5 times, 1st time run w depth = 5, 2nd time run w depth = 4, etc until d is 1
    for d in depth:
        run fwd search for d = 5,4,3,2,1
    '''

    final_policy, best_score, best_comb_parks = forward_search(S, d, U, A, T, R, gamma, prob_dist, best_score, best_comb_parks)[0]

    outputfilename = 'fwd_search.policy'
    write_policy(final_policy, outputfilename)  # TODO: see what policy output looks like, manipulate as needed to be [0 0 1 0 1 ....] of len |S|

    print("time elapsed: {:.2f}s".format(time.time() - start_time))
    return

fwd_search()





'''
def recursive_fwd_search(best_score, prob_dist, S):
    for t in range(100):  # run for 100 seconds/timesteps
        # TODO: On each timestep, choose one position to place a park
        position = random.randint(0, 100)  # TODO: NO WRONG, fwd search will do this by choosing an action
        while S[position][1] == 1:  # if already a park there, choose another one
            position = random.randint(0, 100)
        S[position][1] = 1  # place park there, change second index of tuple to 1
        S[position][2] = 0

        # TODO: Expose to flooding, how to simulate given prob_dist? output booleans (yes flooded or no didnt flood) for each cell
        prob_dist[position] = 0  # need to input last component of all S's in simulate below
        F = simulate(prob_dist, S)
        score = sum(F)

        best_comb_parks = []

        # TODO: Recurse to next subtree, keeping track of best score overall

        # TODO****: make it like k2 graph search, where you add a park then calculate score then remove it? implement fwd search like in book?!

    # TODO: Return best tree path (i.e. optimal placement of 5 parks)
    return best_score
'''