from torch_geometric.nn import MessagePassing
import torch
from torch import nn
from transformers import BertConfig, BertModel
from torch.nn.parameter import Parameter
from torch.nn.modules import Module
import math
from dataset_InsightModel import *
from GraphModel import *
from KAN import KANLinear

class GNNConv(MessagePassing):
    def __init__(self, dim):
        super(GNNConv, self).__init__(node_dim=0, aggr='add')
        self.dim = dim

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.dim)

class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, output_size)
        )

    def forward(self, x):
        x = self.net(x)
        return x

class TextCNN(nn.Module):
    def __init__(self, channels, kernel_size, pool_size, dim, method='max'):
        super(TextCNN, self).__init__()
        self.net1 = nn.Sequential(
            # in_channels,out_channels,kernel_size
            nn.Conv2d(channels, channels, kernel_size[0]),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.MaxPool2d(pool_size)
        )
        self.net2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size[1]),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, dim))
        )
        if method is 'max':
            self.pool = nn.AdaptiveMaxPool2d((1, dim))
        elif method is 'mean':
            self.pool = nn.AdaptiveAvgPool2d((1, dim))
        else:
            raise ValueError('method {} not exist'.format(method))

    def forward(self, x):
        x = self.net1(x)
        x = self.net2(x).squeeze(2)
        x = self.pool(x).squeeze(1)
        return x

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

class GPJFNNF(nn.Module):
    def __init__(self):
        super(GPJFNNF, self).__init__()
        self.user = 256
        self.hidden = 256
        self.config = BertConfig.from_pretrained('./bert-base-chinese/config.json')
        self.model = BertModel.from_pretrained('./bert-base-chinese/pytorch_model.bin', config=self.config)
        self.embd = torch.nn.Embedding(self.user,self.hidden)
        self.emb = nn.Embedding(num_embeddings=4278,embedding_dim=100)
        self.gcn = GraphConvolution(256,256)
        self.gnn_R = SessionGraph(config,config['n_R_node'])
        self.linear = nn.Linear(in_features=200+768,out_features=768)
        self.inputdim = 3072
        self.fc_dim = 512
        self.n_classes = 2
        self.classifier = nn.Sequential(
            nn.Linear(self.inputdim, self.fc_dim),
            nn.Linear(self.fc_dim, self.fc_dim),
            nn.Linear(self.fc_dim, self.n_classes)
        )
        self.kan_linear = KANLinear(in_features=200+768,out_features=768)

    def get_embedding_vec(self, output, mask):
        # torch.Size([16, 64, 768])
        token_embedding = output[0]
        input_mask_expanded = mask.unsqueeze(-1).expand(token_embedding.size()).float()
        sum_embedding = torch.sum(token_embedding * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embedding / sum_mask

    def forward(self, s1_input_ids, s2_input_ids,edge,tgt_batch,zong_ids):
        inputs, mask, len_max = data_masks(edge, [0])
        inputs = np.asarray(inputs)
        mask = np.asarray(mask)
        num = len(tgt_batch)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if inputs[i][j] == 0:
                    mask[i][j] = 0

        alias_inputs, A, items, mask1 = get_slice(inputs,mask,tgt_batch)
        items = trans_to_cuda(torch.Tensor(items).long())
        alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
        A = trans_to_cuda(torch.Tensor(A).float())
        mask = trans_to_cuda(torch.Tensor(mask).long())
        zong_1 = torch.tensor([np.array(f) for f in zong_ids], dtype=torch.float32).view(A.shape[0],A.shape[1],768)
        zong_2 = trans_to_cuda(torch.Tensor(zong_1).float())
        # getR1 = lambda i: zong_1[i][alias_inputs[i]]
        # zong_2 = torch.stack([getR1(i) for i in torch.arange(len(alias_inputs)).long()])
        # s3_mask = torch.ne(zong_1, 0)
        # s3_output = self.model(input_ids=zong_1, attention_mask=s3_mask)
        # s3_output = s3_output[1]
        # s3_output = s3_output.view(edge.shape[0],edge.shape[1],768)
        # getR1 = lambda i: zong_2[i][alias_inputs[i]]
        # s3_output = torch.stack([getR1(i) for i in torch.arange(len(alias_inputs)).long()])

        hidden_R = self.gnn_R(items, A)
        getR = lambda i: hidden_R[i][alias_inputs[i]]
        seq_hidden2 = torch.stack([getR(i) for i in torch.arange(len(alias_inputs)).long()])
        H_global_R = self.gnn_R.compute_globalhidden(seq_hidden2, mask,num)
        s1_mask = torch.ne(s1_input_ids, 0)
        s2_mask = torch.ne(s2_input_ids, 0)

        s1_output = self.model(input_ids=s1_input_ids, attention_mask=s1_mask)
        s2_output1 = self.model(input_ids=s2_input_ids, attention_mask=s2_mask)

        # torch.Size([16, 768])
        s1_vec = s1_output[1]
        s2_vec = s2_output1[1]
        s2_vec = torch.cat([s2_vec, H_global_R], 1)

        # s2_vec = torch.cat(s2_vec,s2_output2)

        # s2_vec = self.kan_linear(s2_vec)
        s2_vec = self.linear(s2_vec)
        # s1_vec = self.get_embedding_vec(s1_output, s1_mask)
        # s2_vec = self.get_embedding_vec(s2_output, s2_mask)

        # features = torch.cat((s1_vec, s2_vec, torch.abs(s1_vec - s2_vec), s1_vec * s2_vec), 1)
        # output = self.classifier(features)

        return s1_vec, s2_vec