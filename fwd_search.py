'''
FWD SEARCH: takes in 10x10 grid, loops through all actions (a la fwd search) to find placement of 5 parks that
maximizes the sum of rewards (aka minimizes the number of grids that flood)
'''

import random
import math
import numpy as np
import time
from scipy.stats import multivariate_normal
import matplotlib.pylab as plt
import seaborn as sns


# Writes final policy to a final .policy file (as we did in project 2), and includes the final score on the last line.
# This policy is unused for our purposes, but is included for possible future use to compare policies given larger
# datasets.
def write_policy(final_policy, final_score, filename):
    with open(filename, 'w') as f:
        for a in final_policy:  # TODO: check if policy is list, change if not
            f.write("{}\n".format(a))
        f.write("Final Score: {}\n".format(final_score))


# Returns a random probability distribution of flooding across the 10x10 grid (performed more poorly than gaussian,
# which is why we ultimately chose to implement this with a gaussian prob dist, which may be more similar to actual
# flooding data (i.e. given topology)).
def random_probs(num_states):
    probs = np.zeros(num_states)
    for s in range(num_states):
        probs[s] = random.random()
    return probs


# Returns a uniform probability distribution of flooding across the 10x10 grid (performed more poorly than gaussian)
def uniform_probs(num_states):
    return [.5]*num_states


# Returns a gaussian probability distribution of flooding across the 10x10 grid (performed the best so is the one we
# ultimately used for our model)
def gaussian_probs(num_states):
    mean = [5, 5]
    cov = [[5, 0], [0, 5]]
    dist = multivariate_normal(mean, cov)
    weights = []
    for i in range(10):  # Calculate pdf for all grid points
        for j in range(10):
            weights.append(dist.pdf([i, j])/.03)  # NOTE: added .03 to scale probs so that prob in center was = .9
    return weights


# Returns the probability of flooding for each cell (from a gaussian distribution, as explained above)
def get_probabilities(num_states):
    return gaussian_probs(num_states) # NOTE: tried uniform and random as well but performed less well, mention in paper


# Rollout: randomly adds four more parks to grid
def rollout(S, A, park_limit, prob_dist, num_rollout_parks):
    rollout_parks = random.sample(list(A), num_rollout_parks)
    temp = []
    for a in rollout_parks:
        S[a][0] = 1  # Place park there, change second index of tuple to 1
        temp.append(prob_dist[a])
        prob_dist[a] = 0

        # # defining bounds
        # left_bound = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
        # right_bound = [9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
        # top_bound = [S[:10]]
        # bottom_bound = [S[90:]]
        # north, south, west, east = a - 10, a + 10, a - 1, a + 1
        #
        # if north > 0: prob_dist[north] = 0
        # if south < 99: prob_dist[south] = 0
        # if west not in right_bound and west >= 0: prob_dist[west] = 0
        # if east not in left_bound and east <= 99: prob_dist[east] = 0

    return rollout_parks, temp, S, prob_dist


# Restores the four random parks changed by the rollout back to their original states
def restore(rollout_parks, temp, S, prob_dist):
    # Restore state so can choose next action
    for i in range(len(rollout_parks)):
        a = rollout_parks[i]
        S[a][0] = 0
        prob_dist[a] = temp[i]
    return S, prob_dist


#
def simulate(prob_dist, S, A, R, park_limit, num_rollout_parks):
    # Rollout: pick 4 parks, set prob_dist to 0 for all of them
    rollout_parks, temp, S, prob_dist = rollout(S, A, park_limit, prob_dist, num_rollout_parks)
    # NOTE: to make this better, we could incorporate distance into the algorithm, i.e. if we're close to park then
    # divide probability of flooding by 2 (mention in 'future directions' section of paper)
    for s in range(len(S)):
        if random.random() < prob_dist[s]:
            S[s][1] = 1
            R[s] = -10  # If flooded, reward negatively
            # NOTE: could try different values for the reward here (possible future direction)
        else:
            S[s][1] = 0
            R[s] = 10  # # If not flooded, reward positively
            # NOTE: could try different values for the reward here (possible future direction)
    score = sum(R)
    S, prob_dist = restore(rollout_parks, temp, S, prob_dist)
    return score, R, S, prob_dist


# Runs forward search, as implemented in chapter 15. Main differences:
# 1. uses sum of rewards (after simulating flooding given the placement of that park + the rollout) as a metric for
# finding the best action, rather than calculating the utility vis the lookahead function (note that this means we
# don't use our transition function)
# 2. Loops forward search 5 times (i.e. since we're adding 5 parks) in order to return the best 5 parks to add
def forward_search(S, depth, A, T, R, gamma, prob_dist, best_scores, best_comb_parks, park_limit):
    num_rollout_parks = depth - 1
    for d in range(depth):
        best = [None, -math.inf]  # (a, u)
        for a in A:
            # Simulate Flooding
            S[a][0] = 1  # Place park there, change second index of tuple to 1
            temp = prob_dist[a]
            prob_dist[a] = 0  # Change prob of flooding to zero

            score, R, S, prob_dist = simulate(prob_dist, S, A, R, park_limit, num_rollout_parks)

            # Save if best
            if score > best[1]:  # NOTE: Write about this in project: experimented with different metrics, decided optimizing sum of rewards was better for our project than optimizing utility
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
        num_rollout_parks = num_rollout_parks - 1
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
    A = np.arange(0, num_states)
    for i in range(num_states):  # Initialize states as (whether it has a park there, whether it flooded or not)
        S.append([0, random.random() < prob_dist[i]])
    R = np.zeros(num_states)
    T = np.zeros([num_states, num_states, num_states])
    U = np.zeros(num_states)  # NOTE: in our final version, this is unused since we don't calculate utility
    gamma = 0.95  # NOTE: in our final version, this is unused since we don't calculate utility/use the bellman equation

    best_scores = []
    best_comb_parks = []

    best_comb_parks, best_scores, prob_dist, S = forward_search(S, depth, A, T, R, gamma, prob_dist, best_scores, best_comb_parks, park_limit)
    final_policy = []
    for a in A:
        if a in best_comb_parks:
            final_policy.append(1)
        else:
            final_policy.append(0)

    final_score, R, S, prob_dist = simulate(prob_dist, S, A, R, park_limit)
    # final_score = sum(best_scores)/len(best_scores)  # NOTE: alt way to calculate final score (get avg score), write about in final paper

    # Draw heap map
    x, y = [], []  # Convert parks to coordinates
    for p in best_comb_parks:
        x.append(int(p // 10) + 0.5)
        y.append(p % 10 + 0.5)
    R_map = [R[0:10], R[10:20], R[20:30], R[30:40], R[40:50], R[50:60], R[60:70], R[70:80], R[80:90], R[90:100]]
    ax = sns.heatmap(R_map, linewidth=0.5)
    ax.scatter(x, y, marker='*', color='red')
    plt.title('Fwd Search Policy: Final Score = ' + str(final_score))
    plt.savefig('fwd_search')

    outputfilename = 'fwd_search.policy'
    write_policy(final_policy, final_score, outputfilename)  # TODO: see what policy output looks like, manipulate as needed to be [0 0 1 0 1 ....] of len |S|

    print("time elapsed: {:.2f}s".format(time.time() - start_time))
    return

fwd_search()



# NOTE lookahead, transition, fwd_search commented out below bc mykel said we didnt need to use them,
# implemented simpler version above
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

