import random
import sys
import requests
import io
import pandas as pd
import numpy as np
import math
import time


def get_data():
    # TODO: edit this filename if needed/add other files
    raw_url = 'https://raw.githubusercontent.com/rozyeastaugh/cs238_finalproject/main/cs238_finalproject_sarsdata.csv'
    download = requests.get(raw_url).content
    df = pd.read_csv(io.StringIO(download.decode('utf-8')))
    return df


def get_states_actions(data):
    S = np.array(data['s'])
    # S = np.arange(1, 312020)
    A = np.array(data['a'])
    return S, A


# Sets all policies to 0 except for 5 random positions on the grid, which is sets to 1 (add park)
def get_policy(S, A, park_limit):
    S.sort()
    final_policy = {key: 0 for key in S}
    parks = random.sample(list(S), park_limit)
    for p in parks:
        final_policy[p] = 1
    return final_policy


def write_policy(p_dict, filename):
    with open(filename, 'w') as f:
        for a in p_dict.values():
            f.write("{}\n".format(a))


park_limit = 5
outputfilename = 'random.policy'
start_time = time.time()
data = get_data()
S, A = get_states_actions(data)
final_policy = get_policy(S, A, park_limit)
write_policy(final_policy, outputfilename)
