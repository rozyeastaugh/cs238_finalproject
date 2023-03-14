'''
This file was used to return both the dummy policy and random policy, to compare to the fwd search policy

DUMMY: takes in 10x10 grid, sets placement of parks to be first 5 cells (first 5 rows of column 0), calculates sum of
rewards

RANDOM: takes in 10x10 grid, adds 5 random placement of parks, calculates sum of rewards
'''


import random
import numpy as np
import math
import time
import matplotlib.pylab as plt
import seaborn as sns
from scipy.stats import multivariate_normal


# Sets all policies to 0 except for 5 random positions on the grid, which is sets to 1 (add park)
def get_policy(S, A, park_limit, parks):
    S.sort()
    final_policy = {key: 0 for key in S}
    for p in parks:
        final_policy[p] = 1
    return final_policy


def write_policy(final_policy, final_score, filename):
    with open(filename, 'w') as f:
        for a in final_policy.values():
            f.write("{}\n".format(a))
        f.write("Final Score: {}\n".format(final_score))


def simulate(prob_dist, S, R):
    # TODO: pick 4 parks, set prob_dist to 0 for all of them
    for s in range(len(S)):
        if random.random() < prob_dist[s]:
            # S[s][1] = 1
            R[s] = -10  # If flooded, reward negatively TODO: try diff values here
        else:
            # S[s][1] = 0
            R[s] = 10  # # If not flooded, reward poistively TODO: try diff values here
    score = sum(R)
    return score, R, S


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



park_limit = 5
start_time = time.time()
num_states = 100
S = np.arange(0, num_states)
A = np.arange(0, num_states)
# best_comb_parks = [0, 1, 2, 3, 4]
best_comb_parks = random.sample(list(A), park_limit)  # TODO: try this too
final_policy = get_policy(S, A, park_limit, best_comb_parks)

# TO COMPARE TO FWD SEARCH
x, y = [], []  # Convert parks to coordinates
for p in best_comb_parks:
    x.append(int(p // 10) + 0.5)
    y.append(p % 10 + 0.5)
R = np.zeros(num_states)
prob_dist = get_probabilities(num_states)
parks_indices = np.where(np.array(list(final_policy.values())) == 1)
for p in list(parks_indices[0]):
    prob_dist[p] = 0
final_score, R, S = simulate(prob_dist, S, R)
R_map = [R[0:10], R[10:20], R[20:30], R[30:40], R[40:50], R[50:60], R[60:70], R[70:80], R[80:90], R[90:100]]
ax = sns.heatmap(R_map, linewidth=0.5)
ax.scatter(x, y, marker='*', color='red')
plt.title('Random Policy: Final Score = ' + str(final_score))

plt.savefig('random')
write_policy(final_policy, final_score, 'random.policy')
