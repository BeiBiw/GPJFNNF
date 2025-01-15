from torch_geometric.nn import MessagePassing
import torch
from torch import nn
from transformers import BertConfig, BertModel
from torch.nn.parameter import Parameter
from torch.nn.modules import Module
import math
from dataset_InsightModel import *
from GraphModel import *
from model.KAN import KANLinear
import gzip
import pickle
from torch.autograd import Variable
import pandas as pd
import torch
import time
import numpy as np
from tqdm import tqdm
from torch import nn
from config import set_args
from model import SentenceBert
from torch.utils.data import TensorDataset, DataLoader
from utils import compute_corrcoef, l2_normalize
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from utils import *
from dataset_InsightModel import *

class BERTModel(nn.Module):
    def __init__(self):
        super(BERTModel, self).__init__()
        self.user = 256
        self.hidden = 256
        self.config = BertConfig.from_pretrained('./bert-base-chinese/config.json')
        self.model = BertModel.from_pretrained('./outputs/best_pytorch_model.bin', config=self.config)

    def get_embedding_vec(self, output, mask):
        token_embedding = output[0]
        input_mask_expanded = mask.unsqueeze(-1).expand(token_embedding.size()).float()
        sum_embedding = torch.sum(token_embedding * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embedding / sum_mask

    def forward(self, s2_input_ids):
        s2_mask = torch.ne(s2_input_ids,0)
        s2_vec = self.model(input_ids=s2_input_ids, attention_mask=s2_mask)
        s2_vec = s2_vec[1]
        return s2_vec

args = set_args()
with gzip.open(args.train_data_path, 'rb') as f:
    train_features = pickle.load(f)

with gzip.open(args.test_data_path, 'rb') as f:
    test_features = pickle.load(f)

zong_ids = torch.tensor(
        [f for f in [f.s2_input_ids for f in train_features] + [f.s2_input_ids for f in test_features]],
        dtype=torch.long)
model = BERTModel()

bs = 8

# 计算需要的分块数
num_chunks = (zong_ids.size(0) + bs - 1) // bs
batches = torch.chunk(zong_ids, num_chunks)

a = []
for batch in batches:
    output = model(batch)
    for line in output:
        a.append(line.detach().tolist())
    print(output.shape)  # 打印模型输出的形状
with gzip.open('./w2v/features_intial.pkl.gz', 'wb') as fout:
    pickle.dump(a, fout)
