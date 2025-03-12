import torch
import numpy as np


class sumtree_func(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.T_node = buffer_size * 2 - 1
        self.tree = np.zeros(self.T_node)

    def update(self, ind, priority):
        tree_ind = self.buffer_size - 1 + ind
        delta = priority - self.tree[tree_ind]
        self.tree[tree_ind] = priority
        while tree_ind != 0:
            tree_ind = (tree_ind - 1) // 2
            self.tree[tree_ind] += delta

    def sample(self, N, b_size, beta):
        # roulette wheel selection
        b_ind = np.zeros(b_size, dtype=np.uint32)
        IS_w = torch.zeros(b_size, dtype=torch.float32)
        priority_sum = self.tree[0]
        interval = priority_sum / b_size
        for j in range(b_size):
            low = interval * j
            up = low + interval
            rnd = np.random.uniform(low, up)
            Ind, prio = self.select_func(rnd)
            b_ind[j] = Ind
            IS_w[j] = ((1 / N) * (self.tree[0] / prio)) ** beta
        IS_w = IS_w / max(IS_w)
        return b_ind, IS_w

    def select_func(self, rnd):
        parent = 0
        while True:
            child_l = parent * 2 + 1
            child_r = child_l + 1
            if child_l >= self.T_node:
                T_Index = parent
                priority = self.tree[T_Index]
                break
            else:
                if self.tree[child_l] >= rnd:
                    parent = child_l
                else:
                    rnd -= self.tree[child_l]
                    parent = child_r
        Index = T_Index - self.buffer_size + 1
        return Index, priority

    def p_max(self):
        t_priority = self.tree[self.buffer_size - 1:]
        return max(t_priority)
