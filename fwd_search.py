import sys
import requests
import io
import pandas as pd
import random
import math
import numpy as np
import time
from scipy.stats import multivariate_normal
import matplotlib.pylab as plt
import seaborn as sns


def write_policy(final_policy, final_score, filename):
    with open(filename, 'w') as f:
        for a in final_policy:  # TODO: check if policy is list, change if not
            f.write("{}\n".format(a))
        f.write("Final Score: {}\n".format(final_score))


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
            weights.append(dist.pdf([i, j])/.03)  # TODO: added .03 so that prob in center was = .9, almost always floods
    # weights = weights/sum(weights)
    return weights


def get_probabilities(num_states):
    return gaussian_probs(num_states)  # TODO: try uniform and random as well for final paper


def simulate(prob_dist, S, R):
    # TODO: pick 4 parks, set prob_dist to 0 for all of them
    for s in range(len(S)):
        if random.random() < prob_dist[s]:
            S[s][1] = 1
            R[s] = -10  # If flooded, reward negatively TODO: try diff values here
        else:
            S[s][1] = 0
            R[s] = 10  # # If not flooded, reward poistively TODO: try diff values here
    score = sum(R)
    return score, R, S


# TODO: newer version, w/o bellman (or recursion?) - no need for T function? TA said good but check w mykel that this is fine
def forward_search(S, depth, A, T, R, gamma, prob_dist, best_scores, best_comb_parks):
    for d in range(depth):
        best = [None, -math.inf]  # (a, u)
        for a in A:
            # Simulate Flooding
            S[a][0] = 1  # Place park there, change second index of tuple to 1
            temp = prob_dist[a]
            prob_dist[a] = 0  # Change prob of flooding to zero

            score, R, S = simulate(prob_dist, S, R)

            # Save if best
            if score > best[1]:  # NOTE: Write about this in project: experimenting with different metrics; added this cause unsure if ranking by utility is best or if it's better to rank by my more specific score metric; Marc said try both and write about it!
                best = [a, score]

            # Restore state so can choose next action
            S[a][0] = 0  #
            prob_dist[a] = temp
        best_comb_parks.append(best[0])
        best_scores.append(best[1])
        S[best[0]][0] = 1
        prob_dist[best[0]] = 0
        idx = np.where(A == best[0])
        A = np.delete(A, idx)
    return best_comb_parks, best_scores, prob_dist, S


def fwd_search():
    start_time = time.time()

    # Set size of problem
    num_states = 10*10

    # Set limit of number of parks we can add given current budget
    park_limit = 5
    depth = park_limit

    # Define probability distribution for flooding in each of the grid ares
    prob_dist = get_probabilities(num_states)

    # Formulate/Initialize MDP
    S = []
    A = np.arange(0, num_states)  # TODO: check that should start at 1, not 0
    for i in range(num_states): # Initialize states as (whether it has a park there, whether it flooded or not) # NOTE: removed position bc encoded as index of s in S, took out prob of flood bc need deterministic states
        S.append([0, random.random() < prob_dist[i]])
    R = np.zeros(num_states)
    T = np.zeros([num_states, num_states, num_states])
    U = np.zeros(num_states)
    gamma = 0.95  # TODO: play around with diff values

    # Find optimal placement of parks using forward search
    '''
    # NOTE: Marc's other idea for getting all 5 actions was to run fwd search 5 times, 1st time run w depth = 5, 2nd time run w depth = 4, etc until d is 1
    for d in depth:
        run fwd search for d = 5,4,3,2,1
    '''

    best_scores = []
    best_comb_parks = []

    # TODO: bellman version, reinstate if we want
    # final_policy, best_score, best_comb_parks = forward_search_bellman(S, d, U, A, T, R, gamma, prob_dist, best_score, best_comb_parks)[0]

    best_comb_parks, best_scores, prob_dist, S = forward_search(S, depth, A, T, R, gamma, prob_dist, best_scores, best_comb_parks)
    final_policy = []
    for a in A:
        if a in best_comb_parks:
            final_policy.append(1)
        else:
            final_policy.append(0)

    # TODO: ASK; calculate final score which way? do both and compare?
    final_score, R, S = simulate(prob_dist, S, R)
    # final_score = sum(best_scores)/len(best_scores)  # Get average score

    # Draw heap map
    R_map = [R[0:10], R[10:20], R[20:30], R[30:40], R[40:50], R[50:60], R[60:70], R[70:80], R[80:90], R[90:100]]
    ax = sns.heatmap(R_map, linewidth=0.5)
    plt.title('Fwd Search Policy: Final Score = ' + str(final_score))
    plt.savefig('fwd_search')

    outputfilename = 'fwd_search.policy'
    write_policy(final_policy, final_score, outputfilename)  # TODO: see what policy output looks like, manipulate as needed to be [0 0 1 0 1 ....] of len |S|

    print("time elapsed: {:.2f}s".format(time.time() - start_time))
    return

fwd_search()



# TODO: NOTE lookahead, transition, fwd_search commented out below bc mykel said we didnt need to use them,
#  implemented simpler version above
'''
def transition(T, prob_dist, a, s, sp):  # TODO: make T update all states
    T[s, a, sp] = (1 if sp == a else 0, 0 if sp == a else random.random() < prob_dist[s])  # 3D 100x100x100 matrix
    # TODO: this? T[S, a, sp] = [(1 if sp == a else 0, 0 if sp == a else random.random() < prob_dist[s]) for s in S]
    return T  # TODO: make sure T correctly updated
'''

'''
def lookahead(U, s, a, S, T, R, gamma, prob_dist):  # TODO: bad to add prob_dist here?
    # NOTE: Marc's other idea for keeping track of all 5 actions: also return actual U values here... ???
    # TODO: change to s not S? i.e. below code not current code? make U a dict?
    # u_new = []
    # for s in S:
    #     u_new.append(R[s, a] + gamma * sum(transition(T, prob_dist, a, s, sp) * U[sp] for sp in S))
    # return u_new, T
    return R[s, a] + gamma * sum(transition(T, prob_dist, a, s, sp) * U[i] for (i, sp) in enumerate(S)), T
    # return R[s, a] + gamma * sum(transition(T, prob_dist, a, s, sp)*U[sp] for sp in S), T
'''

'''
def forward_search_bellman(s, S, d, U, A, T, R, gamma, prob_dist, best_score, best_comb_parks):
    if d <= 0:
        return (None, U(s))  # (a, u)  # TODO: figure out how to find U(s) given that we're looking at at all S
    best = (None, math.inf)  # (a, u)
    U_prime = forward_search(S, d, U, A, T, R, gamma, prob_dist, best_score, best_comb_parks)[1][1]  # TODO: change this output
    # OLD: U_prime = forward_search(s, d - 1, U)[1]  # recurse, get second index which is u
    for a in A:
        # Simulate Flooding
        S[a][0] = 1  # Place park there, change second index of tuple to 1
        temp = prob_dist[a]
        prob_dist[a] = 0  # Change prob of flooding to zero (TODO: need to input last component of all S's in simulate below??)
        score, R, S = simulate(prob_dist, a, S, R)  # TODO: make sure R and S are correctly updated

        # Get next step of value function
        u, T = lookahead(U_prime, s, a, S, T, R, gamma)  # (a, u)  # TODO: dont input s, just S

        # Save if best  # TODO: this will only save 1 action, not 5, right?!?
        if u > best[1]:  # best[1] = u
            best = (a, u)
            # TODO: add (a, u) to a dictionary, at end sort by u and return top a's?
        if score > best_score:  # NOTE: Write about this in project: experimenting with different metrics; added this cause unsure if ranking by utility is best or if it's better to rank by my more specific score metric; Marc said try both and write about it!
            best_score = score
            best = (a, score)
            # best_comb_parks.append(a)

        # Restore state so can choose next action
        S[a][0] = 0  #
        prob_dist[a] = temp
    best_comb_parks.append(best[0]) # TODO: Check that this correctly adds previously encoded action to current best action in order to return best five actions
    return best, best_score, best_comb_parks
'''


'''
    OR (i dont think so...?)
    S = [[]*num_states]
    U = [[]*num_states]
    A = np.arange(1, num_states)
    for i in range(num_states):
        for i in range(num_states): # Initialize states as (whether it has a park there, whether it flooded or not) # NOTE: removed position bc encoded as index of s in S, took out prob of flood bc need deterministic states
            S[i].append((0, random.random() < prob_dist[i]))
        U.append(np.zeros(num_states))

    R = np.zeros(num_states)
    T = np.zeros([num_states, num_states, num_states])
'''

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

