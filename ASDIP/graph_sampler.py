import numpy as np
from copy import deepcopy
from tqdm import tqdm
import torch
from numba import jit, prange
from numba.typed import Dict, List
import numba

tuple_type = numba.types.Tuple((numba.types.int64, numba.types.int64))
np_float_type = numba.types.float64[:]


@jit(nopython=True)
def fast_get_node_neighbors(nodes, times, max_neighbor_num, node_neighbors, node_neighbor_times):
    sampled_nodes = -np.ones((len(nodes), max_neighbor_num), dtype=np.int32)
    lengths = np.zeros((len(nodes)), dtype=np.int32)
    for i in prange(len(nodes)):
        node = nodes[i]
        time = times[i]
        if node >= 0:
            before = np.searchsorted(node_neighbor_times[node], time, 'left')
            if before <= max_neighbor_num:
                sampled_nodes[i][:before] = node_neighbors[node][:before]
                lengths[i] = before
            else:
                sampled_nodes[i] = node_neighbors[node][:max_neighbor_num]
                lengths[i] = max_neighbor_num
    return sampled_nodes, lengths


class NeighborSampler:
    def __init__(self, users, cascades, times, user_num, cascade_num, max_neighbor_num):
        data = {'user': [[] for _ in range(user_num)],
                'cascade': [[] for _ in range(cascade_num)]}
        self.neighbors = deepcopy(data)
        self.times = deepcopy(self.neighbors)
        self.max_neighbor_num = max_neighbor_num
        self.user_num = user_num
        self.cascade_num = cascade_num

        for user, cascade, time in zip(users, cascades, times):
            data['user'][user].append((cascade, time))
            data['cascade'][cascade].append((user, time))
        for dtype in ['user', 'cascade']:
            m_data = data[dtype]
            for node in range(len(m_data)):
                neigh_data = sorted(m_data[node], key=lambda x: x[1])
                neigh_nodes, neigh_times = zip(*neigh_data)
                self.neighbors[dtype][node].extend(neigh_nodes)
                self.times[dtype][node].extend(neigh_times)
        self.nb_neighbors = self.transdict(self.neighbors, np.int64)
        self.nb_times = self.transdict(self.times, np.float32)

    def transdict(self, data, data_dtype):
        new_data = Dict()
        for dtype in ['user', 'cascade']:
            m_data = data[dtype]
            new_m_data = List([np.array(x, dtype=data_dtype) for x in m_data])
            new_data[dtype] = new_m_data
        return new_data

    def get_cas_neighbors(self, nodes, times):
        sampled_neighbors, lengths = fast_get_node_neighbors(nodes=nodes, times=times,
                                                             max_neighbor_num=self.max_neighbor_num,
                                                             node_neighbors=self.nb_neighbors['cascade'],
                                                             node_neighbor_times=self.nb_times['cascade'])
        return sampled_neighbors, lengths


class WalkSampler:
    def __init__(self, users, cascades, times, user_num, cascade_num, device, causal, sample_num, walk_length,
                 limit_times, dataset, use_saved, seed=0):
        data = {'user': [[] for _ in range(user_num)],
                'cascade': [[] for _ in range(cascade_num)]}
        self.neighbors = deepcopy(data)
        self.times = deepcopy(self.neighbors)
        self.device = device
        self.causal = causal
        self.sample_num = sample_num
        self.dataset = dataset
        self.user_num = user_num
        self.cascade_num = cascade_num
        self.min_time = np.min(times) - 100.0
        self.set_seed(seed)

        for user, cascade, time in zip(users, cascades, times):
            data['user'][user].append((cascade, time))
            data['cascade'][cascade].append((user, time))
        for dtype in ['user', 'cascade']:
            m_data = data[dtype]
            for node in range(len(m_data)):
                neigh_data = sorted(m_data[node], key=lambda x: x[1])
                neigh_nodes, neigh_times = zip(*neigh_data)
                self.neighbors[dtype][node].extend(neigh_nodes)
                self.times[dtype][node].extend(neigh_times)
        self.nb_neighbors = self.transdict(self.neighbors, np.int64)
        self.nb_times = self.transdict(self.times, np.float32)
        self.state = 'train'
        self.limit_times = limit_times
        self.walk_length = walk_length
        self.use_saved = use_saved

    def set_seed(self, seed):
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

    def set_state(self, state):
        assert state == 'train' or state == 'eval'
        self.state = state

    def load_saved_data(self):
        assert self.use_saved == True
        self.use_saved = False
        data = self.generate_saved_data()
        self.use_saved = True
        self.walk_nodes = data['walk_node']

    def generate_saved_data(self):
        all_walk_nodes = []
        batch = 128
        for limit_time in self.limit_times:
            walk_nodes = -torch.ones((self.user_num, self.sample_num, self.walk_length), dtype=torch.int,
                                     device=self.device)
            cnt = 0
            for l in tqdm(range(0, self.user_num, batch), desc=f'generating walks'):
                r = min(l + batch, self.user_num)
                nodes = np.arange(l, r)
                mask = [True if len(self.times['user'][node]) > 0 and self.times['user'][node][0] < limit_time
                        else False for node in nodes]
                if np.sum(mask) > 0:
                    cnt += np.sum(mask)
                    nodes = nodes[mask]
                    times = np.array([100000] * len(nodes))
                    limit_times = np.array([limit_time] * len(nodes))
                    m_walk_nodes = self.get_walks(nodes, times, limit_times)
                    # assert torch.sum(m_walk_nodes[:, :, :2] == -1) == 0
                    walk_nodes[nodes] = m_walk_nodes.to(torch.int)
            print(f"==========={cnt}===============")
            all_walk_nodes.append(walk_nodes)
        data = {'walk_node': all_walk_nodes}
        return data

    def get_walks(self, nodes, timestamps, limit_times):
        if self.use_saved:
            neighbors = -torch.ones((len(nodes), self.sample_num, self.walk_length), dtype=torch.long,
                                    device=self.device)
            for limit_time, s_walk_nodes in zip(
                    self.limit_times,
                    self.walk_nodes):
                mask = limit_time == limit_times
                if np.sum(mask) > 0:
                    neighbors[mask] = s_walk_nodes[nodes[mask]].to(torch.long)
            return neighbors

        seeds = np.array([self.seed] * len(nodes))
        if self.state == 'train':
            seeds = np.array([self.rng.integers(0, 1000000000) for _ in range(len(nodes))])
        timestamps, limit_times = timestamps.astype(np.float32), limit_times.astype(np.float32)
        state = np.random.get_state()
        neighbors, neighbor_times = fast_sample_batch_walks(
            nodes=nodes, timestamps=timestamps, limit_times=limit_times, walk_num=self.sample_num,
            all_times=self.nb_times, all_neighbors=self.nb_neighbors, state=self.state, seeds=seeds,
            causal=self.causal, walk_length=self.walk_length, min_time=self.min_time)
        np.random.set_state(state)
        return torch.tensor(neighbors, dtype=torch.long, device=self.device)

    def transdict(self, data, data_dtype):
        new_data = Dict()
        for dtype in ['user', 'cascade']:
            m_data = data[dtype]
            new_m_data = List([np.array(x, dtype=data_dtype) for x in m_data])
            new_data[dtype] = new_m_data
        return new_data


@jit(nopython=True, parallel=True)
def fast_sample_batch_walks(nodes, timestamps, limit_times, walk_num, all_times, all_neighbors, state,
                            seeds, causal, walk_length, min_time):
    neighbors = -np.ones((len(nodes), walk_num, walk_length), dtype=np.int64)
    neighbors_times = -np.ones((len(nodes), walk_num, walk_length), dtype=np.float32)
    for i in prange(len(nodes)):
        node, timestamp, limit_timestamp = nodes[i], timestamps[i], limit_times[i]
        m_neighbors, m_times = fast_sample_walks(node=node, timestamp=timestamp, limit_time=limit_timestamp,
                                                 walk_num=walk_num, all_times=all_times, all_neighbors=all_neighbors,
                                                 state=state, seed=seeds[i], causal=causal, walk_length=walk_length,
                                                 min_time=min_time)
        neighbors[i] = m_neighbors
        neighbors_times[i] = m_times
    return neighbors, neighbors_times


@jit(nopython=True)
def select_neighbors(neighbors, times, now_time, limit_time, causal):
    before = np.searchsorted(times, limit_time, 'left')
    times = times[:before]
    neighbors = neighbors[:before]

    if len(times) == 0:
        return neighbors, times
    if causal == 'before':
        before = np.searchsorted(times, now_time, 'left')
        neighbors, times = neighbors[:before], times[:before]
        return neighbors, times
    elif causal == 'after':
        after = np.searchsorted(times, now_time, 'right')
        neighbors, times = neighbors[after:], times[after:]
        return neighbors, times
    else:
        return neighbors, times


# sample one walk from a certain node
@jit(nopython=True)
def fast_sample_walks(node, timestamp, limit_time, walk_num, all_times, all_neighbors, state, seed, causal,
                      walk_length, min_time):
    types = ['user', 'cascade']
    walk_nodes, walk_times = -np.ones((walk_num, walk_length), dtype=np.int64), -np.ones((walk_num, walk_length),
                                                                                         dtype=np.float32)
    np.random.seed(seed)
    if state == 'eval':  # to make the sampler related to the node id and node times
        x = np.random.randint(low=0, high=node + 1, size=10)
        x = np.random.randint(low=0, high=int(timestamp) + 1, size=10)
    walk_pos, max_try = 0, (walk_length - 1) * 1000
    while walk_pos < walk_num:
        if causal == 'after':
            now_type_index, now_node, now_time = 0, node, min_time
        elif causal == 'before':
            now_type_index, now_node, now_time = 0, node, limit_time
        elif causal == 'none':
            now_type_index, now_node, now_time = 0, node, min_time + np.random.random() * (limit_time - min_time)
        else:
            assert False
        single_walk_nodes = np.array([now_node] + [-1] * (walk_length - 1), dtype=np.int64)
        single_walk_times = np.array([now_time] + [-1.0] * (walk_length - 1), dtype=np.float32)
        for i in range(walk_length - 1):
            max_try -= 1
            if max_try == 0:
                if walk_pos == 0:
                    walk_nodes[:] = -1
                    walk_times[:] = -1
                else:
                    walk_nodes[walk_pos:] = walk_nodes[walk_pos - 1]
                    walk_times[walk_pos:] = walk_times[walk_pos - 1]
                return walk_nodes, walk_times
            now_type = types[now_type_index]
            m_all_neighbors, m_all_times = select_neighbors(all_neighbors[now_type][now_node],
                                                            all_times[now_type][now_node],
                                                            now_time, limit_time, causal)
            if len(m_all_neighbors) == 0:
                now_node, now_time = -1, -1
                break

            pos = np.random.randint(len(m_all_neighbors))
            # if causal == 'none':
            #     pos = np.random.randint(len(m_all_neighbors))
            # else:
            #     p = np.exp(np.abs(m_all_times - now_time) * temporal_scale)  # log
            #     p /= np.sum(p)
            #     assert np.any(np.isnan(p)) == False
            #     pos = np.searchsorted(np.cumsum(p), np.random.random(), 'right')
            now_time = m_all_times[pos]
            now_node = m_all_neighbors[pos]
            now_type_index = 1 - now_type_index
            single_walk_nodes[i + 1] = now_node
            single_walk_times[i + 1] = now_time
        if now_node == -1:
            continue
        walk_nodes[walk_pos] = single_walk_nodes
        walk_times[walk_pos] = single_walk_times
        walk_pos += 1
    return walk_nodes, walk_times
