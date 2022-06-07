import os

import numpy as np
import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset
)


from numpy.random import randint
from functools import reduce
from numpy.random import rand
import networkx as nx
from os import listdir
from os.path import isfile, join


NUMNODES = 1000
NUMGRAPHS = 100
NUMNODEFEATURES = 1
NUMEDGEFEATURES = 1
NODEDIST = 10.0

TRAINSPLIT = 0.8
VALIDSPLIT = 0.1
TESTSPLIT = 0.1


class Division(InMemoryDataset):
    r"""Test data set that creates random graphs with a fixed number of nodes.
    Each node has one attribute with a real value between 0 and 1 in it.
    The corresponding node label is the value of the attribute divided by ten.
    Args:
        root (string): Root directory where the dataset should be saved.
        split (string): If :obj:`"train"`, loads the training dataset.
            If :obj:`"val"`, loads the validation dataset.
            If :obj:`"test"`, loads the test dataset. (default: :obj:`"train"`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):

        assert split in ['train', 'val', 'test']

        super().__init__(root, transform, pre_transform, pre_filter)

        if split == 'train':
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif split == 'val':
            self.data, self.slices = torch.load(self.processed_paths[1])
        elif split == 'test':
            self.data, self.slices = torch.load(self.processed_paths[2])

    def factors(self, n: int) -> list:
        return set(reduce(list.__add__,
                          ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

    def match(self, a: int, b: int, target: int) -> bool:
        return a * b == target

    def get_random_pair(self, factors: list) -> int:
        a_idx = randint(0, len(factors))
        b_idx = randint(0, len(factors))
        a = factors[a_idx]
        b = factors[b_idx]

        while not self.match(a, b, NUMNODES):
            a_idx = randint(0, len(factors))
            b_idx = randint(0, len(factors))
            a = factors[a_idx]
            b = factors[b_idx]

        return a, b

    @property
    def raw_file_names(self):
        try:
            return [f for f in listdir(self.raw_dir) if isfile(join(self.raw_dir, f))]
        except FileNotFoundError:
            return []

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        facts = list(self.factors(NUMNODES))
        for i in range(NUMGRAPHS):
            u, v = self.get_random_pair(facts)
            G = nx.grid_2d_graph(u, v)

            x = rand(G.number_of_nodes(), NUMNODEFEATURES).astype(np.float32)
            a = nx.to_numpy_array(G, dtype=np.float32)
            y = x / 10
            e = np.full((G.number_of_edges(), NUMEDGEFEATURES), NODEDIST, dtype=np.float32)

            filename = os.path.join(
                self.root, self.raw_dir, f'graph_{i}_{u}x{v}')
            np.savez(filename, x=x, a=a, e=e, y=y)

    def process(self):

        train = []
        valid = []
        test = []



        first_train_idx = 0
        last_train_idx = int(len(self.raw_file_names) * TRAINSPLIT)
        first_valid_idx = last_train_idx + 1
        last_valid_idx = int(len(self.raw_file_names) * VALIDSPLIT) + first_valid_idx
        first_test_idx = last_valid_idx + 1
        last_test_idx = len(self.raw_file_names) - 1
        
        print(f'Creating splits: train: {first_train_idx}:{last_train_idx}, valid: {first_valid_idx}:{last_valid_idx}, test: {first_test_idx}:{last_test_idx}')

        train = self.raw_file_names[first_train_idx:last_train_idx]
        valid = self.raw_file_names[first_valid_idx:last_valid_idx]
        test = self.raw_file_names[first_test_idx:last_test_idx]

        splits = [train, valid, test]

        for i, split in enumerate(splits):

            data_list = []

            for f in split:
                f = np.load(os.path.join(self.root, self.raw_dir, f),
                            allow_pickle=True)
                G = nx.from_numpy_matrix(f['a'])

                x = torch.from_numpy(f['x']).to(torch.float32)
                # better let torch figure out the data type by itself here
                edge_index = torch.tensor(list(G.edges)).t().contiguous()
                edge_attr = torch.from_numpy(f['e']).to(torch.float32)
                y = torch.from_numpy(f['y']).to(torch.float32)

                d = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                         y=y, num_nodes=NUMNODES, num_node_features=NUMNODEFEATURES)
                data_list.append(d)

            if self.pre_filter is not None:
                data_list = [
                    data for data in data_list if self.pre_filter(data)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[i])

            data_list.clear()
