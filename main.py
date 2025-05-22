import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import time
from arguments import arg_parse
from data.utils_data import get_dataset
from data.graph_aug import AugTransform
from models.model import *
from utils.utils import get_logger, setup_seed, save_model, load_model
import copy
from torch_scatter import scatter_sum


def drop_node(data, p=0.2):
    mask = torch.rand(size=(data.num_nodes, ), device=data.x.device)
    mask = (mask > p)
    if data.batch is not None:
        batch = data.batch
        while (scatter_sum(mask.int(), batch) == 0).sum() > 0:
            mask = torch.rand(size=(data.num_nodes,), device=data.x.device)
            mask = (mask > p)
    else:
        while torch.sum(mask.int()) == 0:
            mask = torch.rand(size=(data.num_nodes,), device=data.x.device)
            mask = (mask > p)
    subgraph = data.subgraph(mask)
    return subgraph


@torch.no_grad()
def test(loader, model, eval_mode=True):
    if eval_mode:
        model.eval()
    else:
        model.train()

    total_correct = 0
    for data in loader:
        data = data.to(device)
        _, x_proj = model(data.x, data.edge_index, data.batch, data.num_graphs)
        pred = model.classifier(x_proj).argmax(dim=-1)
        total_correct += int((pred == data.y).sum())
    return total_correct / len(loader.dataset)


@torch.no_grad()
def eval_train(loader, model):
    model.eval()

    total_correct = 0
    for data_dict in loader:
        data = data_dict.to(device)
        _, x_proj = model(data.x, data.edge_index, data.batch, data.num_graphs)
        pred = model.classifier(x_proj).argmax(dim=-1)
        total_correct += int((pred == data.y).sum())
    return total_correct / len(loader.dataset)


def run(seed):
    logger.info('seed:{}'.format(seed))
    epochs = args.epochs
    eval_interval = args.eval_interval
    sdfa_eval_interval = args.sdfa_eval_interval
    log_interval = args.log_interval
    batch_size = args.batch_size
    lr = args.lr
    DS = args.DS
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', DS)
    source_save_path = osp.join('.', 'ckpt', f'{DS}-{args.source_index}.pth')
    target_save_path = osp.join('.', 'ckpt', f'{DS}-{args.source_index}-{args.target_index}.pth')

    logger.info(f'{DS}-{args.source_index}-{args.target_index} PSEUDO-LABEL')

    dataset, (source_train_dataset, source_val_dataset, target_train_dataset, target_test_dataset) = get_dataset(DS, path, args)

    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    source_train_loader = DataLoader(source_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    source_val_loader = DataLoader(source_val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    target_train_loader = DataLoader(target_train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    target_test_loader = DataLoader(target_test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    train_transforms = AugTransform(args.aug)
    print('Calculating uniform targets...')
    criterion = nn.CrossEntropyLoss()

    # print(len(dataset))
    dataset_num_features = source_train_dataset[0].x.shape[1]
    dataset_num_classes = len(set([data.y.item() for data in dataset]))
    print(f'num_features: {dataset_num_features}')
    setup_seed(seed)

    model = GNN(dataset_num_features, args.hidden_dim, args.num_gc_layers, dataset_num_classes, args, device).to(device)
    discriminator = Discriminator(embed_dim=8).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)

    best_val_acc = 0.0
    final_test_acc = 0.0
    raw_target_acc = 0.0
    model_for_tta = None

    if args.cache_pretrain and osp.exists(source_save_path):
        model = load_model(source_save_path, model)
        train_acc = eval_train(source_train_loader, model)
        val_acc = test(source_val_loader, model)
        test_acc = test(target_test_loader, model)
        best_val_acc = val_acc
        raw_target_acc = test_acc
        model_for_tta = copy.deepcopy(model)
    else:
        for epoch in range(1, epochs + 1):
            time_start = time.time()
            loss_all = 0
            model.train()

            dataloader = source_train_loader

            # Train on Source Domain 
            for data_dict in dataloader:
                data = data_dict.to(device)
                optimizer.zero_grad()
                _, x_proj = model(data.x, data.edge_index, data.batch, data.num_graphs)
                pred = model.classifier(x_proj)
                loss = criterion(pred, data.y)

                loss.backward()

                loss_all += loss.item() * data.num_graphs
                optimizer.step()

            if epoch % eval_interval == 0:
                model.eval()
                train_acc = eval_train(dataloader, model)
                val_acc = test(source_val_loader, model)
                test_acc = test(target_test_loader, model)
                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    raw_target_acc = test_acc
                    model_for_tta = copy.deepcopy(model)
                    save_model(source_save_path, model)
                print(f'Epoch: {epoch:03d}, Loss: {loss_all / len(dataloader):.2f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
                    f'Test: {test_acc:.4f}')

    log_text = 'best_val_acc: {:.2f}, final_test_acc: {:.2f}'.format(best_val_acc * 100, raw_target_acc * 100)
    logger.info(log_text)
    print(log_text)


    model = copy.deepcopy(model_for_tta)
    # model = model_for_tta
    if args.method == 'ours':
        def sinkhorn(scores, eps=0.05, n_iters=10):  # B x num_prototypes
            Q = torch.exp(scores / eps).T  # Kx B
            Q = Q / torch.sum(Q)
            K, B = Q.shape
            r = torch.ones(K, device=scores.device) / K
            c = torch.ones(B, device=scores.device) / B
            for _ in range(n_iters):
                Q *= (r / Q.sum(dim=1)).unsqueeze(1)
                Q *= (c / Q.sum(dim=0)).unsqueeze(0)
            return (Q / Q.sum(dim=0, keepdims=True)).T
        
        original_prototypes = model.prototypes.weight.detach().clone()
        shared_threshold = args.base_shared_threshold
        surrogate_loss_memory = torch.zeros(len(target_train_dataset), device=device)

        target_train_dataset = [Data(x=graph.x, edge_index=graph.edge_index, y=graph.y, conf_mask=1) for graph in target_train_dataset]

        target_epochs = args.target_epochs
        selection_interval = 5
        best_val_acc = 0.0
        final_test_acc = 0.0

        params_to_update = []
        # assert args.use_bn
        for name, param in model.named_parameters():
            # if 'encoder.bns' in name or 'proj_head.1' in name or 'classifier_net.1' in name or 'prototypes' in name:
            if 'encoder' not in name:
                params_to_update.append(param)
        optimizer = torch.optim.Adam(params_to_update, lr=args.tta_lr)
        # breakpoint()
        from models.model import DiscriminatorMI
        discriminator_mi = DiscriminatorMI(embed_dim=model.cls_dim).to(device)
        optimizer_mi = torch.optim.Adam(discriminator_mi.parameters(), lr=1e-3)

        for epoch in range(1, target_epochs + 1):
            if epoch % selection_interval == 1:
                confidences = []
                model.eval()
                target_dataloader_eval = DataLoader(target_train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
                with torch.no_grad():
                    for data in target_dataloader_eval:
                        data = data.to(device)
                        _, x_proj = model(data.x, data.edge_index, data.batch, data.num_graphs)
                        pred = model.classifier(x_proj)
                        conf = torch.softmax(pred, dim=1).max(dim=1).values
                        confidences.append(conf)
                confidences = torch.cat(confidences, dim=0)
                corrected_confidences = confidences - surrogate_loss_memory * args.omega
                threshold = torch.kthvalue(corrected_confidences, int(len(corrected_confidences) * (1 - shared_threshold))).values
                for i, graph in enumerate(target_train_dataset):
                    graph.conf_mask = 0 if corrected_confidences[i] < threshold else 1
                shared_threshold -= 0.02

            target_dataloader = DataLoader(target_train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

            loss_sh = 0
            loss_pp = 0
            loss_mi = 0
            loss_all = 0
            model.train()
            surrogate_losses = []
            
            for data in target_dataloader:
                data = data.to(device)
                conf_mask = data.conf_mask
                optimizer.zero_grad()
                loss = 0
                _, x_proj = model(data.x, data.edge_index, data.batch, data.num_graphs)
                x_emb = model.classifier_net(x_proj)  # B x cls_dim
                prototypes = model.prototypes.weight  # C x cls_dim
                if args.update_sinkhorn:
                    sim_mat = F.normalize(x_emb, dim=1) @ F.normalize(prototypes, dim=1).T  # B x C
                    sh_prob = sinkhorn(sim_mat.detach(), eps=args.sinkhorn_eps, n_iters=5)  # B x C
                else:
                    model_for_tta.eval()
                    with torch.no_grad():
                        _, x_proj_tta = model_for_tta(data.x, data.edge_index, data.batch, data.num_graphs)
                        x_emb_tta = model_for_tta.classifier_net(x_proj_tta)
                        sim_mat = F.normalize(x_emb_tta, dim=1) @ F.normalize(prototypes, dim=1).T
                        sh_prob = sinkhorn(sim_mat.detach(), eps=args.sinkhorn_eps, n_iters=5)
                pred = model.prototypes(x_emb)
                prob = torch.softmax(pred, dim=-1)
                dist = torch.cdist(x_emb, prototypes)
                loss_sinkhorn = torch.mean(torch.sum(sh_prob * dist.pow(2), dim=1) * conf_mask)
                loss_prototype_prior = torch.mean((model.prototypes.weight - original_prototypes).pow(2))

                data_new = copy.deepcopy(data)
                data_new = drop_node(data_new, p=args.dropnode_prob)
                _, x_proj_new = model(data_new.x, data_new.edge_index, data_new.batch, data_new.num_graphs)
                x_emb_new = model.classifier_net(x_proj_new)  # B x cls_dim
                discriminator_pos_score = discriminator_mi(x_emb, x_emb_new)
                rand_perm = torch.randperm(x_emb.shape[0], device=x_emb.device)
                x_emb_neg = x_emb_new[rand_perm]
                discriminator_neg_score = discriminator_mi(x_emb, x_emb_neg)

                discriminator_pos_score = torch.clamp(discriminator_pos_score, -10, 10)
                discriminator_neg_score = torch.clamp(discriminator_neg_score, -10, 10)
                
                def softplus(x):
                    return torch.log(1 + torch.exp(x))
                
                loss_mi_by_graph = softplus(-discriminator_pos_score) + softplus(discriminator_neg_score)
                loss_mutual_information = loss_mi_by_graph.mean()
                
                loss = loss_sinkhorn * args.sinkhorn_weight + loss_prototype_prior * args.prior_weight + loss_mutual_information * args.mi_loss_weight
                loss.backward()
                optimizer.step()
                optimizer_mi.step()
                loss_all += loss.item()
                loss_sh += loss_sinkhorn.item()
                loss_pp += loss_prototype_prior.item()
                loss_mi += loss_mutual_information.item()
                surrogate_losses.append(loss_mi_by_graph.detach())


            surrogate_losses = torch.cat(surrogate_losses, dim=0)
            surrogate_loss_memory = (1-args.beta) * surrogate_loss_memory + args.beta * surrogate_losses
            
            if epoch % sdfa_eval_interval == 0:
                model.eval()
                val_acc = test(target_test_loader, model)
                test_acc = test(target_test_loader, model)
                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    final_test_acc = test_acc
                    save_model(target_save_path, model)
                print(f'SFDA Epoch: {epoch:03d}, Test: {test_acc:.4f}', end=' ')
                print(f'loss_all: {loss_all/len(target_dataloader):.4f}', end=' ')
                print(f'loss_sh: {loss_sh/len(target_dataloader):.4f}', end=' ')
                print(f'loss_pp: {loss_pp/len(target_dataloader):.4f}', end=' ')
                print(f'loss_mi: {loss_mi/len(target_dataloader):.4f}')
    else:
        raise ValueError('method not supported')    
    
    return raw_target_acc, final_test_acc


if __name__ == '__main__':
    args = arg_parse()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger = get_logger(args)

    acc_list = []
    raw_acc_list = []

    for seed in range(args.st_seed, args.st_seed+args.number_of_run):
        raw_acc, test_acc = run(seed)
        acc_list.append(test_acc)
        raw_acc_list.append(raw_acc)

    acc_mean = np.mean(acc_list)
    acc_std = np.std(acc_list)
    logger.info("acc_mean: {:.2f}, acc_std: {:.2f}".format(acc_mean * 100, acc_std * 100))
    raw_acc_mean = np.mean(raw_acc_list)
    raw_acc_std = np.std(raw_acc_list)
    logger.info("raw_acc_mean: {:.2f}, raw_acc_std: {:.2f}".format(raw_acc_mean * 100, raw_acc_std * 100))

    #put acc_mean & acc_std into a file
    with open( "./csv/" + args.DS + "_accs_" +args.log_file.replace("txt","csv") ,'a') as f:
        f.write(f'{args.source_index}->{args.target_index},{raw_acc_mean*100},{raw_acc_std*100},{acc_mean*100},{acc_std*100}\n')

