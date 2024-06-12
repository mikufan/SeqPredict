import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import math
from tqdm import tqdm
import sys
import numpy as np
import data_utils


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_seq_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        pe = torch.zeros(max_seq_len + 2, embedding_dim)
        position = torch.arange(0, max_seq_len + 2, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * \
                             (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


def patch_attention(m):
    forward_orig = m.forward

    def wrap(*args, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False

        return forward_orig(*args, **kwargs)

    m.forward = wrap


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out[1])

    def clear(self):
        self.outputs = []


class SeqPredictModel(nn.Module):
    def __init__(self, embedding_dim, num_heads, mlp_dim, n_class, max_len, dropout_rate, n_layer, device):
        super().__init__()
        self.embedding = nn.Embedding(28, embedding_dim)
        self.device = device
        self.max_len = max_len
        self.transformer_encoder = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads, dropout=dropout_rate, batch_first=True
        )
        self.transformer_encoding = nn.TransformerEncoder(encoder_layer=self.transformer_encoder, num_layers=n_layer)
        self.position_encoding = PositionalEncoding(embedding_dim, self.max_len)
        self.hidden = nn.Linear(embedding_dim, mlp_dim)
        self.fc_out = nn.Linear(mlp_dim, n_class)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.logger = data_utils.get_logger('data/output/log/train_log.log')

    def forward(self, seq_input, seq_target, is_train=True):
        seq_input = torch.stack(seq_input)
        seq_input = seq_input.to(self.device)
        seq_target = seq_target.to(self.device)
        embedded = self.embedding(seq_input)
        embedded = self.position_encoding(embedded)
        output = self.transformer_encoding(embedded)
        output = torch.mean(output, dim=1)
        output = self.relu(output)
        output = self.hidden(output)
        output = self.relu(output)
        logits = self.fc_out(output)
        if is_train:
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits, seq_target)
            return loss
        else:
            pred = self.softmax(logits)
            return pred

    def model_training(self, train_data_loader, valid_data_loader, n_epochs, model_path):

        self.logger.info("Model constructed.")
        self.logger.info("Start training ...")
        best_result = 0.0
        for n in range(n_epochs):
            self.train()
            n_batch = len(train_data_loader)
            total_loss = 0.0
            for data_batch in tqdm(train_data_loader, mininterval=2, desc=' -Tot it %d' % n_batch,
                                   leave=True, file=sys.stdout):
                batch_seq = [d.idx_seq for d in data_batch]
                batch_target = [d.label for d in data_batch]
                batch_target = torch.tensor(batch_target)
                self.optimizer.zero_grad()
                batch_loss = self.forward(batch_seq, batch_target)
                batch_loss.backward()
                self.optimizer.step()
                total_loss += batch_loss
            self.logger.info("Loss in epoch " + str(n) + ": " + str(total_loss.cpu().detach().numpy() / n_batch))
            n_valid_batch = len(valid_data_loader)
            correct_count = 0
            res = []
            self.eval()
            n_sample_count = 0
            for valid_data_batch in tqdm(valid_data_loader, mininterval=2, desc=' -Tot it %d' % n_valid_batch,
                                         leave=True, file=sys.stdout):
                n_sample_count += len(valid_data_batch)
                valid_batch_seq = [d.idx_seq for d in valid_data_batch]
                valid_batch_target = [d.label for d in valid_data_batch]
                valid_batch_target = torch.tensor(valid_batch_target)
                preds = self.forward(valid_batch_seq, valid_batch_target, False)
                prediction = np.argmax(preds.cpu().detach().numpy(), axis=1)
                prediction = prediction.flatten()
                target_array = valid_batch_target.cpu().detach().numpy().flatten()
                for i in range(len(prediction)):
                    res.append(prediction[i])
                    if prediction[i] == target_array[i]:
                        correct_count += 1
            self.logger.info(
                "Accuracy in epoch " + str(n) + ": " + str(round(float(correct_count) / n_sample_count * 100, 4)) + "%")
            pred_accuracy = float(correct_count) / n_sample_count * 100
            if pred_accuracy > best_result:
                best_result = pred_accuracy
                torch.save(self, model_path + "/trained_model.pt")

    def model_test(self, test_data_loader):
        n_batch = len(test_data_loader)
        correct_count = 0
        pred_res = []
        true_label = []
        self.eval()
        n_sample_count = 0
        all_attn = []
        save_output = SaveOutput()
        # patch_attention(self.transformer_encoding.layers[0].self_attn)
        # hook_handle = self.transformer_encoding.layers[0].self_attn.register_forward_hook(save_output)
        for test_data_batch in tqdm(test_data_loader, mininterval=2, desc=' -Tot it %d' % n_batch,
                                    leave=True, file=sys.stdout):
            n_sample_count += len(test_data_batch)
            test_batch_seq = [d.idx_seq for d in test_data_batch]
            test_batch_target = [d.label for d in test_data_batch]
            test_batch_target = torch.tensor(test_batch_target)
            attn_weights = self.probe_attention(test_batch_seq, 0)
            all_attn.append(attn_weights.cpu().detach().numpy())
            preds = self.forward(test_batch_seq, test_batch_target, False)
            prediction = np.argmax(preds.cpu().detach().numpy(), axis=1)
            prediction = prediction.flatten()
            target_array = test_batch_target.cpu().detach().numpy().flatten()
            for i in range(len(prediction)):
                true_label.append(target_array[i])
                pred_res.append(prediction[i])
                if prediction[i] == target_array[i]:
                    correct_count += 1
        print("Accuracy in test set: " + str(round(float(correct_count) / n_sample_count * 100, 4)) + "%")
        # self.logger.info("Accuracy in test set: " + str(round(float(correct_count) / n_sample_count * 100, 4)) + "%")
        # all_attn = np.stack(all_attn)
        # print(res)
        # attn_output = save_output.outputs
        return pred_res, true_label, all_attn

    def probe_attention(self, test_seq, layer_idx):
        probe_layer = self.transformer_encoding.layers[layer_idx]
        probe_self_attn = probe_layer.self_attn
        test_seq = torch.stack(test_seq)
        test_seq = test_seq.to(self.device)
        test_seq = self.embedding(test_seq)
        test_seq = self.position_encoding(test_seq)
        _, attention_weights = probe_self_attn(test_seq, test_seq, test_seq)
        return attention_weights
