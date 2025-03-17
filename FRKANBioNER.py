# 作者 ： 李叶霖
# 开发日期 ： 2024/10/8

import torch
import torch.nn as nn
from transformers import PreTrainedModel, AutoModel, AutoConfig
import math
import numpy as np
from einops import rearrange
import torch.nn.functional as F

from fftKAN import NaiveFourierKANLayer


class SLabLinearAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                focusing_factor=3, kernel_size=5):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)
        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, dim)))

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv.unbind(0)
        k = k + self.positional_encoding[:, :C]

        kernel_function = nn.ReLU()
        q = kernel_function(q)
        k = kernel_function(k)

        q, k, v = (rearrange(x, "b n (h c) -> (b h) n c", h=self.num_heads) for x in [q, k, v])

        with torch.cuda.amp.autocast(enabled=False):
            q = q.to(torch.float32)
            k = k.to(torch.float32)
            v = v.to(torch.float32)

            z = 1 / (torch.einsum("b i c,b c->b i", q, k.sum(dim=1)) + 1e-6)
            kv = torch.einsum("b j c, b j d -> b c d", k, v)
            x = torch.einsum("b i c, b c d, b i -> b i d", q, kv, z)

        feature_map = rearrange(v, "b n c -> b c n 1")
        feature_map = rearrange(self.dwc(feature_map), "b c n 1 -> b n c")
        x = x + feature_map

        x = rearrange(x, "(b h) n c -> b n (h c)", h=self.num_heads)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class BiLSTM(nn.Module):
    def __init__(self, hidden_size):
        super(BiLSTM, self).__init__()
        self.forward_lstm = nn.LSTM(hidden_size, hidden_size // 2, num_layers=1, bidirectional=False, batch_first=True)
        self.backward_lstm = nn.LSTM(hidden_size, hidden_size // 2, num_layers=1, bidirectional=False, batch_first=True)

    def forward(self, x):
        batch_size, max_len, feat_dim = x.shape
        out1, _ = self.forward_lstm(x)
        reverse_x = torch.flip(x, [1])
        out2, _ = self.backward_lstm(reverse_x)

        output = torch.cat((out1, out2), 2)
        return output, (1, 1)

class FRKANBioNER(nn.Module):
    def __init__(self, args, num_labels, hidden_dropout_prob=0.1, windows_list=None):
        super(FRKANBioNER, self).__init__()
        config = AutoConfig.from_pretrained(args.bert_model)
        self.bert = AutoModel.from_pretrained(args.bert_model)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.num_labels = num_labels
        self.use_bilstm = args.use_bilstm
        self.use_multiple_window = args.use_multiple_window
        self.windows_list = windows_list
        self.connect_type = args.connect_type
        self.d_model = args.d_model
        self.slab = SLabLinearAttention(dim=self.d_model, window_size=(self.d_model, self.d_model), num_heads=8)

###
        # # 新增 NaiveFourierKANLayer，作为 BERT 层后的处理
        # self.fourier_kan = NaiveFourierKANLayer(inputdim=self.d_model, outdim=self.d_model, gridsize=3)
###

        if self.use_multiple_window:
            if self.use_bilstm:
                self.bilstm_layers = nn.ModuleList([BiLSTM(self.d_model) for _ in self.windows_list])
            else:
                self.bilstm_layers = nn.ModuleList([nn.LSTM(self.d_model, self.d_model, num_layers=1, bidirectional=False, batch_first=True) for _ in self.windows_list])

        # self.linear = nn.Linear(self.d_model, self.num_labels)

        # # 使用 nn.Sequential 合并两个 MLP 层
        # self.classifier = nn.Sequential(
        #     nn.Linear(self.d_model, 50),
        #     nn.Linear(50, self.num_labels)
        # )

        # 将分类头替换为 NaiveFourierKANLayer
        self.classifier = NaiveFourierKANLayer(inputdim=self.d_model, outdim=self.num_labels, gridsize=3)

        # # 使用 nn.Sequential 合并两个 KAN 层
        # self.classifier = nn.Sequential(
        #     NaiveFourierKANLayer(inputdim=self.d_model, outdim=16, gridsize=5),
        #     NaiveFourierKANLayer(inputdim=16, outdim=self.num_labels, gridsize=5)
        # )

    def windows_sequence(self, sequence_output, windows, lstm_layer):
        batch_size, max_len, feat_dim = sequence_output.shape
        local_final = torch.zeros([batch_size, max_len, feat_dim], dtype=torch.float32, device='cuda')
        for i in range(max_len):
            index_list = [j for j in range(max(0, i - windows // 2), min(max_len, i + windows // 2 + 1))]
            temp = sequence_output[:, index_list, :]
            out, _ = lstm_layer(temp)
            local_final[:, i, :] = out[:, -1, :]
        return local_final

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None, attention_mask_label=None):
        sequence_output = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]
        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32, device='cuda')

        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if valid_ids[i][j].item() == 1:
                    jj += 1
                    valid_output[i][jj] = sequence_output[i][j]
        sequence_output = self.dropout(valid_output)

###
        # # 添加 NaiveFourierKANLayer 层处理
        # sequence_output = self.fourier_kan(sequence_output)
###

        if self.use_multiple_window:
            mutiple_windows = []
            for i, window in enumerate(self.windows_list):
                local_final = self.windows_sequence(sequence_output, window, self.bilstm_layers[i])
                mutiple_windows.append(local_final)
            sequence_output = sum(mutiple_windows)

            if self.connect_type == 'dot-att':
                muti_local_features = torch.stack(mutiple_windows, dim=2)
                sequence_output = sequence_output.unsqueeze(dim=2)
                d_k = sequence_output.size(-1)
                attn = torch.matmul(sequence_output, muti_local_features.permute(0, 1, 3, 2)) / math.sqrt(d_k)
                attn = torch.softmax(attn, dim=-1)
                local_features = torch.matmul(attn, muti_local_features).squeeze()
                sequence_output = sequence_output.squeeze()
                sequence_output = sequence_output + local_features
            elif self.connect_type == 'mlp-att':
                mutiple_windows.append(sequence_output)
                muti_features = torch.cat(mutiple_windows, dim=-1)
                muti_local_features = torch.stack(mutiple_windows, dim=2)
                query = self.Q(muti_features)
                d_k = query.size(-1)
                query = query.unsqueeze(dim=2)
                attn = torch.matmul(query, muti_local_features.permute(0, 1, 3, 2)) / math.sqrt(d_k)
                attn = torch.softmax(attn, dim=-1)
                sequence_output = torch.matmul(attn, muti_local_features).squeeze()
            elif self.connect_type == 'slab-att':
                sequence_output = self.slab(sequence_output)


        # logits = self.linear(sequence_output)
        # 使用 NaiveFourierKANLayer 作为分类头
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            if attention_mask_label is not None:
                active_loss = attention_mask_label.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

