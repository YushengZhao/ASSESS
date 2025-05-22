import torch

def get_splits_in_domain(dataset, ratio=0.8):
    data_idx = torch.randperm(len(dataset))
    if type(dataset) is list:
        train_dataset = []
        test_dataset = []
        for i in range(len(dataset)):
            if i < int(len(dataset) * ratio):
                train_dataset.append(dataset[data_idx[i]])
            else:
                test_dataset.append(dataset[data_idx[i]])
        return train_dataset, test_dataset
    dataset = dataset[data_idx]
    num_data = len(data_idx)
    num_train = int(num_data * ratio)
    train_index = data_idx[:num_train]
    test_index = data_idx[num_train:]

    train_dataset = dataset[train_index]
    test_dataset = dataset[test_index]

    return train_dataset, test_dataset


def get_domain_splits(dataset, split=4):
    try:
        dataset.data.y = dataset.data.y.long()
    except:
        print('dataset.data does not have y attribute')
    # return get_domain_splits_nonuni(dataset, split)
    info = {}  # data info of each class
    for i in range(len(dataset)):
        if dataset[i].num_nodes < 2:
            continue
        density = dataset[i].num_edges / (dataset[i].num_nodes * (dataset[i].num_nodes - 1))
        key = dataset[i].y.cpu().item()
        if key not in info:
            info[key] = [(i, density)]
        else:
            info[key].append((i, density))

    indices = {i: [] for i in range(split)}  # data info of each split
    for key in info:
        info[key].sort(key=lambda x: x[1])
        cls_len = len(info[key])
        for i in range(split):
            indices[i].extend([info[key][j][0] for j in range(i * (cls_len // split), (i + 1) * (cls_len // split))])
    
    return_dataset = []
    for key in indices:
        indices[key].sort()
        if type(dataset) == list:
            return_dataset.append([dataset[i] for i in indices[key]])
        else:
            indices[key] = torch.tensor(indices[key])
            return_dataset.append(dataset[indices[key]])

    return return_dataset


def get_domain_splits_nonuni(dataset, split=4):
    node_density = []
    for i in range(len(dataset)):
        if dataset[i].num_nodes < 2:
            continue
        node_density.append(dataset[i].num_edges / (dataset[i].num_nodes * (dataset[i].num_nodes - 1)))
    node_density = torch.tensor(node_density)
    node_density, data_idx = torch.sort(node_density, descending=False)

    return_dataset = []
    for i in range(split):
        return_dataset.append(dataset[data_idx[i*(len(dataset) // split) : (i+1)*(len(dataset) // split)]])

    return return_dataset
