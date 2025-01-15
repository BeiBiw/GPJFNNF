import os
import gzip
import pickle
from torch.autograd import Variable
import pandas as pd
import torch
import time
import numpy as np
from tqdm import tqdm
from torch import nn
# from config import set_args
from model.GPJFNNF import GPJFNNF
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from dataset_InsightModel import *

def l2_normalize(vecs):
    """标准化
    """
    norms = (vecs**2).sum(axis=1, keepdims=True)**0.5
    return vecs / np.clip(norms, 1e-8, np.inf)

CUDA_LAUNCH_BLOCKING=1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weight = torch.FloatTensor(2).fill_(1)
loss_fn = nn.CrossEntropyLoss()
# loss_fn.size_average = False

class Features:
    def __init__(self, s1_input_ids=None, s2_input_ids=None, label=None):
        self.s1_input_ids = s1_input_ids
        self.s2_input_ids = s2_input_ids
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "s1_input_ids: %s" % (self.s1_input_ids)
        s += ", s2_input_ids: %s" % (self.s2_input_ids)
        s += ", label: %d" % (self.label)
        return s



def evaluate(model):
    model.eval()
    # 语料向量化
    all_a_vecs, all_b_vecs = [], []
    all_labels = []
    all_pred =[]
    for step, batch in tqdm(enumerate(test_dataloader)):
        s1_input_ids, s2_input_ids, edge,label = batch
        if torch.cuda.is_available():
            s1_input_ids = s1_input_ids.cuda()
            s2_input_ids = s2_input_ids.cuda()
            label = label.cuda()
        batch_ids2 = []
        inputs, mask, len_max = data_masks(edge, [0])
        inputs = np.asarray(inputs)
        mask = np.asarray(mask)
        num = len(label)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if inputs[i][j] == 0:
                    mask[i][j] = 0

        alias_inputs, A, items, mask1 = get_slice(inputs, mask, label)

        for i in range(len(items)):
            for j in range(len(items[0])):
                if items[i][j] != 0:
                    batch_ids2.append(intial_features[items[i][j]])
                else:
                    batch_ids2.append(torch.tensor([0] * 768, dtype=torch.float32))

        with torch.no_grad():
            s1_embeddings, s2_embeddings = model(s1_input_ids,s2_input_ids,edge,label,batch_ids2)
            s1_embeddings = s1_embeddings.cpu().numpy()
            s2_embeddings = s2_embeddings.cpu().numpy()
            label = label.cpu().numpy()
            #
            all_a_vecs.extend(s1_embeddings)
            all_b_vecs.extend(s2_embeddings)
            all_labels.extend(label)


            # output = model(s1_input_ids, s2_input_ids, edge, label)
            # pred = output.data.max(1)[1]
            #
            # label = label.cpu().numpy()
            #
            # all_labels.extend(label)
            # all_pred.extend(pred.cpu().numpy())
            # loss = loss_fn(output, label)


            # acc_train = accuracy(pred, label)

    all_a_vecs = np.array(all_a_vecs)
    all_b_vecs = np.array(all_b_vecs)
    all_labels = np.array(all_labels)
    #
    a_vecs = l2_normalize(all_a_vecs)
    b_vecs = l2_normalize(all_b_vecs)
    #
    sims = (a_vecs * b_vecs).sum(axis=1)
    pred = []
    for i in sims:
        if i>=0.5:
            pred.append(1)
        if i<0.5:
            pred.append(0)

    acc = accuracy_score(all_labels,pred)
    precison = precision_score(all_labels,pred)
    recall = recall_score(all_labels,pred)
    f1 = f1_score(all_labels,pred)
    return acc,precison,recall,f1


def calc_loss(s1_vec, s2_vec, true_label):
    loss_fct = nn.MSELoss()
    output = torch.cosine_similarity(s1_vec, s2_vec)
    loss = loss_fct(output, true_label)
    return loss

if __name__ == '__main__':
    args = set_args()
    args.output_dir = 'outputs'
    os.makedirs(args.output_dir, exist_ok=True)

    with gzip.open(args.train_data_path, 'rb') as f:
        train_features = pickle.load(f)

    with gzip.open(args.test_data_path, 'rb') as f:
        test_features = pickle.load(f)

    with gzip.open(args.feature_intial, 'rb') as f:
        intial_features = pickle.load(f)

    train_edge_path = os.path.join(args.edge_path,'train_edge1.txt')  ##构建一个完整的路径
    test_edge_path = os.path.join(args.edge_path,'test_edge1.txt')

    train_edge = []
    with open(train_edge_path, 'r') as file:
        for line in file:
            line1 = line.strip().split(',')
            line2 = [int(num) for num in line1]
            train_edge.append(line2)

    # test_edge = test_edge['0']
    test_edge = []
    with open(test_edge_path, 'r') as file:
        for line in file:
            line1 = line.strip().split(',')
            line2 = [int(num) for num in line1]
            test_edge.append(line2)
    # 开始训练
    print("***** Running training *****")
    print("  Num examples = {}".format(len(train_features)))
    print("  Batch size = {}".format(args.train_batch_size))
    # g1 = graphData(train_edge)
    # g2 = graphData(test_edge)
    def data_mask(all_usr_pois, item_tail):
        us_lens = [len(upois) for upois in all_usr_pois]
        len_max = max(us_lens)
        us_pois = [pos + [item_tail]*(len_max-len(pos)) for pos in all_usr_pois]

        return us_pois

    train_edge = data_mask(train_edge,0)
    test_edge = data_mask(test_edge,0)

    train_s1_input_ids = torch.tensor([f.s1_input_ids for f in train_features], dtype=torch.long)
    train_s2_input_ids = torch.tensor([f.s2_input_ids for f in train_features], dtype=torch.long)
    train_label_ids = torch.tensor([f.label for f in train_features], dtype=torch.float32)

    g1 = torch.tensor([f for f in train_edge], dtype=torch.long)

    train_data = TensorDataset(train_s1_input_ids, train_s2_input_ids,g1, train_label_ids)  ###创建了一个 TensorDataset 对象，将这些张量组合在一起
    train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True)  ###创建了一个 DataLoader 对象，用于批量加载数据

    test_s1_input_ids = torch.tensor([f.s1_input_ids for f in test_features], dtype=torch.long)
    test_s2_input_ids = torch.tensor([f.s2_input_ids for f in test_features], dtype=torch.long)
    test_label_ids = torch.tensor([f.label for f in test_features], dtype=torch.float32)

    g2 = torch.tensor([f for f in test_edge], dtype=torch.long)


    test_data = TensorDataset(test_s1_input_ids, test_s2_input_ids, g2,test_label_ids)
    test_dataloader = DataLoader(test_data, batch_size=args.train_batch_size, shuffle=True)
    zong_ids = torch.tensor([f for f in [f.s2_input_ids for f in train_features] + [f.s2_input_ids for f in test_features]],dtype=torch.long)

    num_train_steps = int(
         len(train_features) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)  ###整个训练过程需要的梯度回传（优化）次数
        ###gradient_accumulation_steps（梯度累积步数）：训练多少个batch后进行一次梯度回传（适用于显存受限的情况下进行更大批次的训练）

    # 模型
    model = GPJFNNF()  ##实例化model

    # 指定多gpu运行
    if torch.cuda.is_available():  ###检查是否有GPU，如果有则将model传到GPU上
        model.cuda()

    tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese/config.json')   ###从本地路径加载预训练的 BERT tokenizer 配置（Tokenizer 是 BERT 模型的重要组成部分，用于将文本转换为模型可以处理的张量形式）

    ###准备优化器参数
    param_optimizer = list(model.named_parameters())   ###列出所有模型参数
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']  ###不进行权重衰减的参数

    optimizer_grouped_parameters = [  ###将参数分为俩组，一组进行权重衰减，一组不进行
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    ###配置优化器和学习率调度器
    warmup_steps = 0.05 * num_train_steps  ###计算学习率预热步数，为总训练步数的 5%
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)  ###使用 AdamW 优化器，并传入分组的参数列表、学习率（args.learning_rate）和 epsilon（用于数值稳定性的参数）
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_steps   ###使用 get_linear_schedule_with_warmup 函数创建线性学习率调度器，设置预热步数和总训练步数（num_train_steps）
    )
    best_acc = 0   ###初始化用于记录最佳准确率（best_acc）
    all_costs = []
    words_count = 0
    intial_features = torch.tensor([f for f in intial_features],dtype=torch.float32)
    for epoch in range(args.num_train_epochs):

        for step, batch in enumerate(train_dataloader):
            global_step = 0

            start_time = time.time()
            if torch.cuda.is_available():
                batch = tuple(t.cuda() for t in batch)
            s1_input_ids, s2_input_ids,edge, label = batch  ###将 batch 数据解包
            inputs, mask, len_max = data_masks(edge, [0])
            inputs = np.asarray(inputs)
            mask = np.asarray(mask)
            num = len(label)
            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    if inputs[i][j] == 0:
                        mask[i][j] = 0

            alias_inputs, A, items, mask1 = get_slice(inputs, mask, label)

            batch_ids = []
            for i in range(len(items)):
                for j in range(len(items[0])):
                    if items[i][j] != 0:
                        batch_ids.append(intial_features[items[i][j]])
                    else:
                        batch_ids.append(torch.tensor([0] * 768, dtype=torch.float32))

                        # edge_1 = graphData(edge_1)
            s1_vec, s2_vec = model(s1_input_ids, s2_input_ids,edge,label,batch_ids)
            # output = model(s1_input_ids, s2_input_ids,edge,label)
            # pred = output.data.max(1)[1]
            # label = torch.tensor(label, dtype=torch.long).cuda()

            # loss = loss_fn(output, label)
            # all_costs.append(loss.item())
            # words_count += (s1_input_ids.nelement() + s2_input_ids.nelement()) / 200
            # acc_train = accuracy(pred,label)

            loss = calc_loss(s1_vec, s2_vec, label)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # print('Epoch:{}, Step:{}, Loss:{:10f},acc_train:{:4f}, Time:{:10f}'.format(epoch, step, loss, acc_train,time.time() - start_time))
            print('Epoch:{}, Step:{}, Loss:{:10f}, Time:{:10f}'.format(epoch, step, loss,time.time() - start_time))

            sss = 'epoch:{}, loss:{}'.format(epoch,loss)
            with open(args.output_dir + '/loss.txt',"a+",encoding="utf-8") as f:
                sss += '\n'
                f.write(sss)
            loss.backward()  ###进行反向传播，计算梯度

            # nn.utils.clip_grad_norm(model.parameters(), max_norm=20, norm_type=2)   # 是否进行梯度裁剪

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()  ##更新模型参数
                scheduler.step()  ##更新学习率
                optimizer.zero_grad()  ##清零梯度
                global_step += 1
            # 一轮跑完 进行eval
        acc,pre,recall,f1 = evaluate(model)
        print(f"epoch:{epoch},acc:{acc},pre:{pre},recall:{recall},f1:{f1}")

        ss = 'epoch:{},acc:{},pre:{},recall:{},f1:{}'.format(epoch,acc,pre,recall,f1)
        with open(args.output_dir + '/logs.txt', 'a+', encoding='utf8') as f:
            ss += '\n'
            f.write(ss)
        if acc > best_acc:
            print(f"best epoch:\t\n acc:{acc}\t\n pre:{pre} recall:{recall}\t\n f1:{f1}\t\n ")
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self 检查model是否具有module属性。这个属性通常在模型被DataParallel或DistributedDataParallel包裹时存在。
            output_model_file = os.path.join(args.output_dir, "best_pytorch_model.bin")
            torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save = model.module if hasattr(model, 'module') else model
        output_model_file = os.path.join(args.output_dir, "epoch{}_ckpt.bin".format(epoch))
        torch.save(model_to_save.state_dict(), output_model_file)