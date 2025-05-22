from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.utils import degree
import torch
import torch.nn.functional as F
from .data_splits import get_splits_in_domain, get_domain_splits
import numpy as np
import random
from torch_scatter import scatter_mean


class OneHotDegreeClipped(T.BaseTransform):
    def __init__(self, max_degree, in_degree=False, cat=True):
        self.max_degree = max_degree
        self.in_degree = in_degree
        self.cat = cat

    def __call__(self, data):
        idx, x = data.edge_index[1 if self.in_degree else 0], data.x
        deg = degree(idx, data.num_nodes, dtype=torch.long)
        deg = torch.clamp_max(deg, self.max_degree)
        deg = F.one_hot(deg, num_classes=self.max_degree + 1).to(torch.float)

        if x is not None and self.cat:
            x = x.view(-1, 1) if x.dim() == 1 else x
            data.x = torch.cat([x, deg.to(x.dtype)], dim=-1)
        else:
            data.x = deg

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.max_degree})'
    
class EdgeFeaturesToNodeFeatures(T.BaseTransform):
    def __call__(self, data):
        edge_attr = data.edge_attr
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        
        node_features_from_edges = scatter_mean(edge_attr, edge_index[1], dim=0, dim_size=num_nodes)
        
        if data.x is not None:
            data.x = torch.cat([data.x, node_features_from_edges], dim=1)
        else:
            data.x = node_features_from_edges
        
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

class ProcessLabel(T.BaseTransform):
    def __call__(self, data):
        data.y = data.y.view(-1)
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def get_dataset(DS, path, args):
    setup_seed(0)

    if DS in ['COX2', 'BZR']: # cross dataset
        DSS = [DS, f'{DS}_MD']
        source_dataset = TUDataset(path, name=DSS[args.source_index], use_node_attr=True)
        target_dataset = TUDataset(path, name=DSS[args.target_index], use_node_attr=True)
    else:
        if DS in ['IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'REDDIT-MULTI-5K', 'reddit_threads']:
            dataset = TUDataset(path, name=DS, use_node_attr=True, transform=T.Compose([
                OneHotDegreeClipped(64),
                T.AddSelfLoops()
            ]))
        else:
            dataset = TUDataset(path, name=DS, use_node_attr=True)
        print(f'Dataset: {DS}, Length: {len(dataset)}')
        source_split_index = args.source_index
        target_split_index = args.target_index
        split = args.data_split
        split_dataset = get_domain_splits(dataset, split)
        source_dataset = split_dataset[source_split_index]
        target_dataset = split_dataset[target_split_index]
    source_train_dataset, source_val_dataset = get_splits_in_domain(source_dataset)
    target_train_dataset, target_test_dataset = get_splits_in_domain(target_dataset)

    return dataset, (source_train_dataset, source_val_dataset, target_train_dataset, target_test_dataset)
