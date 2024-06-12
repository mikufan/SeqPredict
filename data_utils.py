import os
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import Sampler
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import numpy as np
import seaborn as sns
import logging
from tqdm import tqdm
import sys


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def plot_cm(pred_res, true_label):
    # 计算混淆矩阵
    cm = confusion_matrix(pred_res, true_label)

    # 创建标签列表
    labels = ['2-iminopropanoate-deaminase', 'adenosine-deaminase', 'aminocyclopropane-1-carboxylate-deaminase',
              'AMP-deaminase', 'CMP-deaminase',
              'cytidine-deaminase', 'cytosine-deaminase', 'deoxycytidylate-deaminase',
              'Editase', 'Glucosamine', 'guanine-deaminase', 'guanosine-deaminas',
              'Intermediate-Deaminase-A-chloroplastic-like', 'L-threonine-deaminase', 'porphobilinogen-deaminase']

    # 绘制混淆矩阵
    fig, ax = plt.subplots(figsize=(40, 35))
    # plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    # plt.title('Confusion Matrix')
    # plt.colorbar()
    # tick_marks = np.arange(len(labels))
    # plt.xticks(tick_marks, labels, rotation=45)
    # plt.yticks(tick_marks, labels)
    # plt.xlabel('Predicted Label')
    # plt.ylabel('True Label')
    # sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=labels, yticklabels=labels)
    # # 在每个格子内显示数值
    # thresh = cm.max() / 3.0
    # for i in range(len(labels)):
    #     for j in range(len(labels)):
    #         if cm[i, j] > thresh:
    #             ax.text(j, i, cm[i, j], ha="right", va="center", color="w")
    #         else:
    #             ax.text(j, i, cm[i, j], ha="right", va="center", color="b")
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    ax.set_xlabel('X Label', fontsize=30)
    ax.set_ylabel('Y Label', fontsize=30)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    cm_display.plot(cmap="Blues", ax=ax, xticks_rotation="vertical", text_kw={"fontsize": 20})
    plt.savefig('data/output/figures/confusion_matrix.png')


def attention_stats(attention_weights, test_seq, id2token):
    attention_pair = dict()
    n_sample = len(attention_weights)
    for n in tqdm(range(n_sample), mininterval=2, desc=' -Tot it %d' % n_sample,
                  leave=True, file=sys.stdout):
        _, len_sen, _ = attention_weights[n].shape
        seq_attention_weights = np.squeeze(attention_weights[n], axis=0)
        id_seq = test_seq[n].idx_seq
        max_value_index = np.argmax(seq_attention_weights)
        max_attention_pos = np.unravel_index(max_value_index, seq_attention_weights.shape)
        max_attention_value = seq_attention_weights[max_attention_pos]
        best_ids = (int(id_seq[max_attention_pos[0]].numpy()), int(id_seq[max_attention_pos[1]].numpy()))
        best_pair = (id2token[best_ids[0]], id2token[best_ids[1]])
        if attention_pair.get(best_pair) is not None:
            attention_pair[best_pair] = attention_pair[best_pair] + 1
        else:
            attention_pair[best_pair] = 1
    return attention_pair


def traverse_read(data_folder_path, file_list, name_list):
    for file_name in os.listdir(data_folder_path):
        path = os.path.join(data_folder_path, file_name)
        if os.path.isdir(path):
            traverse_read(data_folder_path, file_list, name_list)
        else:
            file_df = pd.read_csv(path)
            file_list.append(file_df)
            name_list.append(file_name)


def data_split(data, test_size, seed):
    train_indices, split_test_indices, _, _ = train_test_split(
        range(len(data)),
        data.targets,
        stratify=data.targets,
        test_size=test_size,
        random_state=seed
    )
    test_targets = [data.targets[i] for i in split_test_indices]
    test_valid_indices, test_test_indices, _, _ = train_test_split(
        range(len(split_test_indices)),
        test_targets,
        stratify=test_targets,
        test_size=0.5,
        random_state=seed
    )
    valid_indices = [split_test_indices[i] for i in test_valid_indices]
    test_indices = [split_test_indices[i] for i in test_test_indices]
    # generate subset based on indices
    train_split = Subset(data, train_indices)
    valid_split = Subset(data, valid_indices)
    test_split = Subset(data, test_indices)
    return train_split, valid_split, test_split


def get_max_len(data_items):
    max_len = 0
    for d in data_items:
        seq_len = len(d.seq)
        if seq_len > max_len:
            max_len = seq_len
    return max_len


def update_vocab(data_seq, vocab, id_to_token):
    index = len(vocab)
    if type(data_seq) is not float:
        for token in data_seq:
            if token not in vocab:
                vocab[token] = index
                id_to_token[index] = token
                index += 1


class LengthSortSampler(Sampler):
    def __init__(self, data_source):
        super(LengthSortSampler, self).__init__(data_source)
        self.data_source = data_source
        self.sorted_indices = sorted(range(len(data_source)), key=lambda i: len(data_source[i].seq))

    def __iter__(self):
        return iter(self.sorted_indices)

    def __len__(self):
        return len(self.data_source)


def collate_fn(batch):
    sequences = []
    for d in batch:
        idx_tensor = d.get_idx_seq()
        sequences.append(idx_tensor)
    sequences = pad_sequence(sequences, batch_first=True)
    for i in range(len(batch)):
        batch[i].idx_seq = sequences[i]
    return batch


class DataItem(object):
    def __init__(self, id_num, seq, label):
        self.id_num = id_num
        self.seq = seq
        self.label = label
        self.idx_seq = []

    def set_idx_seq(self, vocab):
        self.idx_seq.append(vocab['START'])
        for token in self.seq:
            self.idx_seq.append(vocab[token])
        self.idx_seq.append(vocab['END'])
        self.idx_seq = torch.tensor(self.idx_seq)

    def get_idx_seq(self):
        return self.idx_seq


class SeqDataset(Dataset):
    def __init__(self, data_folder_path, is_curation, label_name):
        self.data_items = []
        self.label_dict = dict()
        self.vocab = dict()
        self.id_2_token = dict()
        self.read_data(data_folder_path, is_curation, label_name)
        self.targets = [d.label for d in self.data_items]
        self.max_len = get_max_len(self.data_items)
        for d in self.data_items:
            d.set_idx_seq(self.vocab)

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        return self.data_items[idx]

    def read_data(self, data_folder_path, is_curation, label_name):
        vocab = {'PAD': 0, 'START': 1, 'END': 2}
        id_2_token = {0: 'PAD', 1: 'START', 2:'END'}
        file_list = []
        file_names = []
        traverse_read(data_folder_path, file_list, file_names)
        label_idx = 0
        if not is_curation:
            for data_file, data_name in zip(file_list, file_names):
                label = os.path.basename(data_name)
                self.label_dict[label] = label_idx
                for data_idx, data_row in data_file.iterrows():
                    data_item = DataItem(data_row['id'], data_row['seqs'], label_idx)
                    self.data_items.append(data_item)
                    update_vocab(data_item.seq, vocab, id_2_token)
                label_idx += 1
        else:
            for data_file in file_list:
                for data_idx, data_row in data_file.iterrows():
                    label = data_row[label_name]
                    if label not in self.label_dict.keys():
                        self.label_dict[label] = label_idx
                        label_idx += 1
                    data_item = DataItem(data_row['Gene ID'], data_row['Full_seq'], self.label_dict[label])
                    self.data_items.append(data_item)
                    update_vocab(data_item.seq, vocab, id_2_token)


        self.vocab = vocab
        self.id_2_token = id_2_token
