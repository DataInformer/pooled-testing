import random

import numpy as np
import itertools
from scipy.special import comb

def generate_pools(num_samples=6, num_pools=3, pools_per_sample=None):
    '''
    :param num_samples: number of samples
    :param num_pools: number of pools
    :param pools_per_sample: number of pools/sample (optional)
    :return: a boolean and list of lists, where each list gives the individuals
            that should be in a pool
    '''

    if pools_per_sample is None:
        pools_per_sample = np.argmax([comb(num_pools, i) for i in range(num_pools)])
    print(pools_per_sample)
    pool_size = int(np.ceil(num_samples / num_pools) * pools_per_sample)
    encodings = list(itertools.combinations(range(num_pools), pools_per_sample))
    num_enc = len(encodings)
    print(encodings, num_enc)
    cur_pool_sizes = np.zeros(num_pools)
    new_pool_sizes = np.zeros(num_pools)
    sample_codes = []
    step = 0
    for n in range(num_samples):
        print(n)
        maxsize = pool_size + 1
        while maxsize > pool_size: # at least 1 iteration
            np.copyto(new_pool_sizes, cur_pool_sizes)
            # print(cur_pool_sizes)
            step += 1
            candidate = encodings[step % num_enc]
            for p in candidate:
                new_pool_sizes[p] += 1
            maxsize = np.max(new_pool_sizes)
            print(maxsize)
        np.copyto(cur_pool_sizes, new_pool_sizes)
        sample_codes.append(candidate)

     # [encodings[i % num_enc] for i in range(num_samples)]
    print(sample_codes)
    pools = {i: [] for i in range(num_pools)}
    for i, code in enumerate(sample_codes):
        for j in code:
            pools[j].append(i)
    return num_enc > num_samples, pools



def disjunct_pooler(num_samples=6, num_pools=3, pools_per_sample=None):
    if pools_per_sample is None:
        pools_per_sample = np.argmax([comb(num_pools, i) for i in range(num_pools)])
    print(pools_per_sample)
    pool_size = int(np.ceil(num_samples / num_pools) * pools_per_sample)
    encodings = list(itertools.combinations(range(num_pools), pools_per_sample))
    num_enc = len(encodings)
    print(encodings, num_enc)
    num_left = num_samples
    sample_codes = []
    while num_left > num_enc:
        sample_codes += encodings
        num_left -= num_enc
    print(sample_codes)
    sample_codes += random.sample(encodings, num_left)
    print(sample_codes)
    '''
    if num_enc >= num_samples:
        sample_codes = np.random.choice(encodings, num_samples, replace=False)
    else:
        print(np.random.choice(encodings, num_samples-num_enc))
        sample_codes = np.r_[encodings, np.random.choice(encodings, num_samples-num_enc, replace=True)]
    '''
    pools = {i: [] for i in range(num_pools)}
    for i, code in enumerate(sample_codes):
        for j in code:
            pools[j].append(i)
    return num_enc > num_samples, pools


def pooler(num_samples=6, num_pools=3, pools_per_sample=None):
    if pools_per_sample is None:
        pools_per_sample = np.argmax([comb(num_pools, i) for i in range(num_pools)])
    pool_size = int(np.ceil(num_samples / num_pools) * pools_per_sample)
    pools = np.zeros((num_samples, num_pools), dtype=bool)
    for repeats in range(pools_per_sample):
        for n in range(num_samples):
            small_pools = np.where(pools.sum(axis=0) == pools.sum(axis=0).min())[0]
            print(small_pools)
            # cand_pool= []
            # print(pools[n][cand_pool] == 0)
            unfound = True
            while unfound:
                cand_pool = random.choice(small_pools)
                if (pools[n][cand_pool] == 0):
                    pools[n][cand_pool] = 1
                    unfound = False
                # print(cand_pool)
    return pools

if __name__ == '__main__':
    print(generate_pools(192,16))