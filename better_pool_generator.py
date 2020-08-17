import numpy as np
from itertools import combinations
from scipy.special import comb
from collections import defaultdict
import random

class PoolGenerator:
    def __init__(self, num_samples, num_pools, pools_per_sample=None):
        self.num_samples = num_samples
        self.num_pools = num_pools
        if pools_per_sample is None:
            self.pools_per_sample = self._get_pps()
        else:
            self.pools_per_sample = pools_per_sample
        # self.pool_size = self.num_samples * self.pools_per_sample / self.num_pools
        self.reset(keep=False)
        self.repeats = np.ceil(self.num_samples /
                               comb(self.num_pools, self.pools_per_sample)).astype(int)
        self.solved = False
        # print(self.repeats)

    def _get_pps(self):
        powers = [comb(self.num_pools, i) for i in range(self.num_pools)]
        for i,pow in sorted(enumerate(powers), key=lambda x:x[1]):
            if pow >= self.num_samples:
                return i
        return i #backup is largest option
        # enough = [i for i in range(self.num_pools) if powers[i] >= self.num_samples ]

    def reset(self, keep = True):
        if not keep:
            self.keepers = []
        self._setup_structs()
        self.pool_counts = np.zeros(self.num_pools, dtype=int)
        self.pool_members = defaultdict(list)
        self.sample_pools = []

    def _setup_structs(self):
        self.combos = list(combinations(range(self.num_pools), self.pools_per_sample))
        self.pool_used = defaultdict(list) #tracks which combo indices include a pool
        for i, c in enumerate(self.combos):
            for j in c:
                self.pool_used[j].append(i)
        self.pool_used = {k:set(v) for k,v in self.pool_used.items()}

    def get_pools(self):
        if not self.solved:
            self._solve()
        self.sample_pools = [self.combos[k] for k in self.keepers]
        for i,pool_list in enumerate(self.sample_pools):
            for p in pool_list:
                self.pool_members[p].append(i)
        return self.pool_members

    def _solve(self):
        num_left = self.num_samples
        if self.repeats > 1:
            csize = comb(self.num_pools, self.pools_per_sample).astype(int)
            # print(csize)
            self.keepers = list(range(csize)) * (self.repeats - 1)
            self.pool_counts += csize * self.pools_per_sample \
                                //self.num_pools * (self.repeats - 1)
            num_left -= csize * (self.repeats - 1)
        for i in range(num_left):
            self._pick_next_keeper()
        self.solved = True

    def _pick_next_keeper(self):
        for gap in range(self.num_pools):  # check conditions as we go; shouldn't reach this high
            cur_pools = get_min_idxs(self.pool_counts, gap=gap)
            #         print(len(cur_pools))
            if len(cur_pools) >= self.pools_per_sample:
                metacombos = list(combinations(cur_pools, self.pools_per_sample))
                random.shuffle(metacombos)
                for combo in metacombos:
                    candidate = set.intersection(*[self.pool_used[i] for i in combo])
                    #                 print(candidate, combo, gap)
                    if len(candidate) > 0:
                        k = candidate.pop()
                        self.keepers.append(k)
                        for j in self.combos[k]:
                            self.pool_counts[j] += 1
                            self.pool_used[j].remove(k)
                        return 0
            elif len(cur_pools) == 2:
                candidates = self.pool_used[cur_pools[0]].intersection(self.pool_used[cur_pools[1]])
                #             print(candidates)
                if candidates:
                    candidate = random.choice(list(candidates))
                    self.keepers.append(candidate)
                    for j in self.combos[candidate]:
                        self.pool_counts[j] += 1
                        self.pool_used[j].remove(candidate)

    def show_stats(self):
        if not self.solved:
            print('Solve first')
            return
        print('We placed each sample in {} pools.\n'.format(self.pools_per_sample))
        print('With this arrangement, we might need up to {} follow up tests for '
              'each pool that is positive\n'.format(0 if self.repeats==1 else self.repeats))
        print('The pool sizes range from {} to {}, with an average of {}'.format(
            self.pool_counts.min(), self.pool_counts.max(), np.mean(self.pool_counts)
        ))

def get_min_idxs(mylist, gap=0):
    minval = min(mylist)
    le = len(mylist)
    return [i for i in range(le) if mylist[i] <= minval + gap]