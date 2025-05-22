import torch
import torch.nn as nn
from .gnn import Encoder
import torch.nn.functional as F


class TTNBatchNorm1d(nn.Module):
    def __init__(self, bn, alpha=0.3):
        super(TTNBatchNorm1d, self).__init__()
        self.bn = bn
        self.alpha = alpha
    
    def forward(self, x):
        self.bn.eval()
        mean = torch.mean(torch.mean(x, dim=0, keepdim=True), dim=-1, keepdim=True)
        var = torch.mean(torch.mean((x - mean) ** 2, dim=0, keepdim=True), dim=-1, keepdim=True)
        bn_mean = self.bn.running_mean
        bn_var = self.bn.running_var
        new_mean = self.alpha * mean + (1 - self.alpha) * bn_mean
        new_var = self.alpha * var + (1 - self.alpha) * bn_var
        weight = self.bn.weight
        bias = self.bn.bias
        eps = self.bn.eps
        x = (x - new_mean) / torch.sqrt(new_var + eps)
        x = x * weight + bias
        return x

class MABNBatchNorm1d(nn.Module):
    def __init__(self, bn):
        super(MABNBatchNorm1d, self).__init__()
        self.bn = bn
        self.weight = nn.Parameter(bn.weight.data.clone())
        self.bias = nn.Parameter(bn.bias.data.clone())
    
    def forward(self, x):
        self.bn.eval()
        bn_mean = self.bn.running_mean
        bn_var = self.bn.running_var
        weight = self.weight
        bias = self.bias
        eps = self.bn.eps
        x = (x - bn_mean) / torch.sqrt(bn_var + eps)
        x = x * weight + bias
        return x


class GNN(nn.Module):
    def __init__(self, dataset_num_features, hidden_dim, num_gc_layers, num_class, args, device):
        super(GNN, self).__init__()

        self.device = device

        self.embedding_dim = hidden_dim
        self.cls_dim = 8
        self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers, device=device, conv_type=args.conv_type,
                               use_bn=args.use_bn, JK=args.JK, global_pool=args.global_pool)

        self.prototypes = nn.Linear(self.cls_dim, num_class, bias=False)  # weight shape: num_class x cls_dim
        if args.use_bn:
            self.proj_head = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.BatchNorm1d(self.embedding_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0),
                nn.Linear(self.embedding_dim, self.cls_dim)
            )
            self.classifier_net = nn.Sequential(
                nn.Linear(self.cls_dim, self.cls_dim),
                nn.BatchNorm1d(self.cls_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0),
                nn.Linear(self.cls_dim, self.cls_dim)
            )
        else:
            self.proj_head = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0),
                nn.Linear(self.embedding_dim, self.embedding_dim)
            )
            self.classifier_net = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0),
                nn.Linear(self.embedding_dim, self.cls_dim)
            )

        self.init_emb()

    def classifier(self, x):
        x = self.classifier_net(x)
        x = self.prototypes(x)
        return x
    
    def init_emb(self):
        # initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, batch, num_graphs):

        if x is None:
            x = torch.ones(batch.shape[0]).to(self.device)

        y = self.encoder(x, edge_index, batch)

        y_proj = self.proj_head(y)

        return y, F.normalize(y_proj, dim=1)

    def loss_cal(self, x, x_aug):

        T = 0.2
        batch_size, _ = x.size()

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug)

        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()

        return loss
    
class GNNWithDropout(GNN):
    def __init__(self, dataset_num_features, hidden_dim, num_gc_layers, num_class, args, device, dropout_rate):
        super().__init__(dataset_num_features, hidden_dim, num_gc_layers, num_class, args, device)

        # Add dropout layer after the projection head
        if args.use_bn:
            self.proj_head = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.BatchNorm1d(self.embedding_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(args.teacher_dropout),
                nn.Linear(self.embedding_dim, self.embedding_dim)
            )
            self.classifier = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.BatchNorm1d(self.embedding_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(args.teacher_dropout),
                nn.Linear(self.embedding_dim, num_class)
            )
        else:
            self.proj_head = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(args.teacher_dropout),
                nn.Linear(self.embedding_dim, self.embedding_dim)
            )
            self.classifier = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(args.teacher_dropouts),
                nn.Linear(self.embedding_dim, num_class)
            )

class Discriminator(nn.Module):
    def __init__(self, embed_dim):
        super(Discriminator, self).__init__()
        self.fc = nn.Linear(embed_dim, 1)
    
    def forward(self, embeddings):
        logits = self.fc(embeddings)
        return torch.sigmoid(logits)
        

class DiscriminatorMI(nn.Module):
    def __init__(self, embed_dim):
        super(DiscriminatorMI, self).__init__()
        self.fc_1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )
        self.fc_2 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def forward(self, embeddings_1, embeddings_2):
        emb_1 = self.fc_1(embeddings_1) + embeddings_1
        emb_2 = self.fc_2(embeddings_2) + embeddings_2
        score = torch.sum(emb_1 * emb_2, dim=1)
        return score