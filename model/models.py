# -*- coding: utf-8 -*-
# @Time    : 2020/8/27 2:52 下午
# @Author  : jeffery (modified by Gemini)
# @FileName: models.py
# @Description:
from base import BaseModel
from transformers import AutoConfig, AutoModel
from torch import nn
from torch.nn import functional as F
import pickle
import torch
import numpy as np
import os
from pathlib import Path
from utils.model_utils import prepare_pack_padded_sequence


# ---
# (Non-Transformer models remain unchanged)
# ---

class FastText(BaseModel):
    def __init__(self, class_num, train, dropout, 
                 vocab_size, embedding_dim, embedding_matrix): 
        super().__init__()
        self.embedding_size = embedding_dim
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix,
                                                      freeze=(not train))
        self.fc = nn.Linear(self.embedding_size, class_num)

    def forward(self, text, _, text_lengths):
        embedded = self.embedding(text).float()
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1)
        return self.fc(pooled), pooled


class TextCNN(BaseModel):
    def __init__(self, n_filters, filter_sizes, dropout, train, class_num,
                 vocab_size, embedding_dim, embedding_matrix): 
        super().__init__()
        self.embedding_size = embedding_dim
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix,
                                                      freeze=(not train))
        
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs, self.embedding_size)) for fs in
             filter_sizes])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, class_num)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, _, text_lengths):
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1).float()
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, int(conv.shape[2])).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat), cat


class TextCNN1d(BaseModel):
    def __init__(self, n_filters, filter_sizes, dropout, train, class_num,
                 vocab_size, embedding_dim, embedding_matrix): 
        super().__init__()
        self.embedding_size = embedding_dim
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix,
                                                      freeze=(not train))
        self.convs = nn.ModuleList(
            [nn.Conv1d(in_channels=self.embedding_size, out_channels=n_filters, kernel_size=fs) for fs in filter_sizes])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, class_num)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, _, text_lengths):
        embedded = self.embedding(text)
        embedded = embedded.permute(0, 2, 1).float()
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        pooled = [F.max_pool1d(conv, int(conv.shape[2])).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat), cat


class RnnModel(BaseModel):
    def __init__(self, rnn_type, hidden_dim, class_num, n_layers, bidirectional, dropout, train,
                 batch_first, vocab_size, embedding_dim, embedding_matrix): 
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.embedding_size = embedding_dim
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix,
                                                      freeze=(not train))

        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.embedding_size,
                               hidden_dim,
                               num_layers=n_layers,
                               bidirectional=bidirectional,
                               batch_first=batch_first,
                               dropout=dropout)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(self.embedding_size,
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)
        else:
            self.rnn = nn.RNN(self.embedding_size,
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)

        self.fc = nn.Linear(hidden_dim * n_layers, class_num)
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

    def forward(self, text, _, text_lengths):
        text, sorted_seq_lengths, desorted_indices = prepare_pack_padded_sequence(text, text_lengths.to('cpu'))
        
        embedded = self.dropout(self.embedding(text)).float()

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_seq_lengths.to('cpu'), batch_first=self.batch_first)
        self.rnn.flatten_parameters()
        if self.rnn_type in ['rnn', 'gru']:
            packed_output, hidden = self.rnn(packed_embedded)
        else:
            packed_output, (hidden, cell) = self.rnn(packed_embedded)

        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=self.batch_first)
        output = output[desorted_indices]
        batch_size, max_seq_len, hidden_dim = output.shape
        hidden = torch.mean(torch.reshape(hidden, [batch_size, -1, hidden_dim]), dim=1)
        output = torch.mean(output, dim=1)
        fc_input = self.dropout(output + hidden)
        out = self.fc(fc_input)

        return out, fc_input


class RCNNModel(BaseModel):
    def __init__(self, rnn_type, hidden_dim, class_num, n_layers, bidirectional, dropout, train,
                 batch_first, vocab_size, embedding_dim, embedding_matrix): 
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding_size = embedding_dim
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix,
                                                      freeze=(not train))

        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.embedding_size,
                               hidden_dim,
                               num_layers=n_layers,
                               bidirectional=bidirectional,
                               batch_first=batch_first,
                               dropout=dropout)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(self.embedding_size,
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)
        else:
            self.rnn = nn.RNN(self.embedding_size,
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)

        self.fc_cat = nn.Linear(hidden_dim * n_layers + self.embedding_size, self.embedding_size)
        self.fc = nn.Linear(self.embedding_size, class_num)
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

    def forward(self, text, _, text_lengths):
        text, sorted_seq_lengths, desorted_indices = prepare_pack_padded_sequence(text, text_lengths.to('cpu'))
        embedded = self.dropout(self.embedding(text)).float()
        
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_seq_lengths.to('cpu'), batch_first=self.batch_first)

        self.rnn.flatten_parameters()
        if self.rnn_type in ['rnn', 'gru']:
            packed_output, hidden = self.rnn(packed_embedded)
        else:
            packed_output, (hidden, cell) = self.rnn(packed_embedded)

        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=self.batch_first)
        output = output[desorted_indices]

        batch_size, max_seq_len, hidden_dim = output.shape
        output = torch.tanh(self.fc_cat(torch.cat((output, embedded), dim=2)))
        output = torch.transpose(output, 1, 2)
        output = F.max_pool1d(output, int(max_seq_len)).squeeze().contiguous()

        return self.fc(output), output


class RnnAttentionModel(BaseModel):
    def __init__(self, rnn_type, hidden_dim, class_num, n_layers, bidirectional, dropout, train,
                 batch_first, vocab_size, embedding_dim, embedding_matrix): 
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.batch_first = batch_first
        
        self.embedding_size = embedding_dim
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix,
                                                      freeze=(not train))

        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.embedding_size,
                               hidden_dim,
                               num_layers=n_layers,
                               bidirectional=bidirectional,
                               batch_first=batch_first,
                               dropout=dropout)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(self.embedding_size,
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)
        else:
            self.rnn = nn.RNN(self.embedding_size,
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)

        self.tanh1 = nn.Tanh()
        self.tanh2 = nn.Tanh()
        self.w = nn.Parameter(torch.randn(hidden_dim), requires_grad=True)

        self.dropout = nn.Dropout(dropout)
        if bidirectional:
            self.w = nn.Parameter(torch.randn(hidden_dim * 2), requires_grad=True)
            self.fc = nn.Linear(hidden_dim * 2, class_num)
        else:
            self.w = nn.Parameter(torch.randn(hidden_dim), requires_grad=True)
            self.fc = nn.Linear(hidden_dim, class_num)

    def forward(self, text, _, text_lengths):
        text, sorted_seq_lengths, desorted_indices = prepare_pack_padded_sequence(text, text_lengths.to('cpu'))
        embedded = self.dropout(self.embedding(text)).to(torch.float32)

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_seq_lengths.to('cpu'), batch_first=self.batch_first)

        if self.rnn_type in ['rnn', 'gru']:
            packed_output, hidden = self.rnn(packed_embedded)
        else:
            packed_output, (hidden, cell) = self.rnn(packed_embedded)

        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=self.batch_first)
        output = output[desorted_indices]
        hidden = hidden[desorted_indices]
        
        alpha = F.softmax(torch.matmul(self.tanh1(output), self.w), dim=0).unsqueeze(-1)
        output_attention = output * alpha

        batch_size, max_seq_len, hidden_dim = output.shape
        hidden = torch.mean(torch.reshape(hidden, [batch_size, -1, hidden_dim]), dim=1)

        output_attention = torch.sum(output_attention, dim=1)
        output = torch.sum(output, dim=1)

        fc_input = self.dropout(output + output_attention + hidden)
        out = self.fc(fc_input)
        return out, fc_input


class DPCNN(nn.Module):
    def __init__(self, n_filters, class_num, train,
                 vocab_size, embedding_dim, embedding_matrix): 
        super(DPCNN, self).__init__()

        self.embedding_size = embedding_dim
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix,
                                                      freeze=(not train))

        self.conv_region = nn.Conv2d(1, n_filters, (3, self.embedding_size), stride=1)
        self.conv = nn.Conv2d(n_filters, n_filters, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom
        self.relu = nn.ReLU()
        self.fc = nn.Linear(n_filters, class_num)

    def forward(self, text, _, text_lengths):
        x = self.embedding(text)
        x = x.unsqueeze(1).to(torch.float32)
        x = self.conv_region(x)

        x = self.padding1(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.padding1(x)
        x = self.relu(x)
        x = self.conv(x)
        while x.size()[2] >= 2:
            x = self._block(x)
        x_embedding = x.squeeze()
        x = self.fc(x_embedding)
        return x, x_embedding

    def _block(self, x):
        x = self.padding2(x)
        px = self.max_pool(x)
        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)
        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)
        x = x + px
        return x


# ---
# (Transformer 模型已修正)
# ---
class TransformersModel(BaseModel):

    def __init__(self, transformer_model, cache_dir, force_download, is_train, class_num):
        super(TransformersModel, self).__init__()
        self.transformer_config = AutoConfig.from_pretrained(transformer_model, cache_dir=cache_dir,
                                                             force_download=force_download)
        self.transformer_model = AutoModel.from_pretrained(transformer_model, config=self.transformer_config,
                                                           cache_dir=cache_dir, force_download=force_download)

        # 是否对transformers参数进行训练
        for name, param in self.transformer_model.named_parameters():
            param.requires_grad = is_train

        # (*** 关键修正 1 ***)
        # 不再使用 .to_dict()['hidden_size']
        self.fc = nn.Linear(self.transformer_model.config.hidden_size, class_num)

    def forward(self, input_ids, attention_masks, text_lengths):
        # (适配 distilbert, 它只返回 last_hidden_state)
        transformer_output = self.transformer_model(input_ids, attention_mask=attention_masks)
        # last_hidden_state = [batch_size, seq_len, hidden_size]
        last_hidden_state = transformer_output[0] 
        # (取 [CLS] token, 它在 distilbert 中是第一个 token)
        cls = last_hidden_state[:, 0, :] 

        out = self.fc(cls)
        return out, cls


class TransformersCNN(nn.Module):

    def __init__(self, transformer_model, cache_dir, force_download, n_filters, filter_sizes, dropout,
                 is_train, class_num):
        super(TransformersCNN, self).__init__()
        self.transformer_config = AutoConfig.from_pretrained(transformer_model, cache_dir=cache_dir,
                                                             force_download=force_download)
        self.transformer_model = AutoModel.from_pretrained(transformer_model, config=self.transformer_config,
                                                           cache_dir=cache_dir, force_download=force_download)

        for name, param in self.transformer_model.named_parameters():
            param.requires_grad = is_train
        
        # (*** 关键修正 2 ***)
        hidden_size = self.transformer_model.config.hidden_size
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, n_filters, (k, hidden_size)) for k in filter_sizes])
        
        self.dropout = nn.Dropout(dropout)
        self.fc_cnn = nn.Linear(n_filters * len(filter_sizes), class_num)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, int(x.size(2))).squeeze(2)
        return x

    def forward(self, input_ids, attention_masks, text_lengths):
        # (适配 distilbert)
        encoder_out = self.transformer_model(input_ids, attention_mask=attention_masks)[0]
        
        encoder_out = self.dropout(encoder_out)
        out = encoder_out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out_embedding = self.dropout(out)
        out = self.fc_cnn(out_embedding)
        return out, out_embedding


class TransformersRNN(nn.Module):

    def __init__(self, transformer_model, cache_dir, force_download, rnn_type, hidden_dim, n_layers, bidirectional,
                 batch_first, dropout,is_train, class_num):
        super(TransformersRNN, self).__init__()

        self.transformer_config = AutoConfig.from_pretrained(transformer_model, cache_dir=cache_dir,
                                                             force_download=force_download)
        self.transformer_model = AutoModel.from_pretrained(transformer_model, config=self.transformer_config,
                                                           cache_dir=cache_dir, force_download=force_download)

        for name, param in self.transformer_model.named_parameters():
            param.requires_grad = is_train

        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.batch_first = batch_first

        # (*** 关键修正 3 ***)
        hidden_size = self.transformer_model.config.hidden_size
        
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(hidden_size,
                               hidden_size=hidden_dim,
                               num_layers=n_layers,
                               bidirectional=bidirectional,
                               batch_first=batch_first,
                               dropout=dropout)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(hidden_size,
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)
        else:
            self.rnn = nn.RNN(hidden_size,
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.fc_rnn = nn.Linear(hidden_dim * 2, class_num)

    def forward(self, input_ids, attention_masks, text_lengths):
        # (适配 distilbert)
        sentence_out = self.transformer_model(input_ids, attention_mask=attention_masks)[0]

        # 按照句子长度从大到小排序
        bert_sentence, sorted_seq_lengths, desorted_indices = prepare_pack_padded_sequence(sentence_out, text_lengths.to('cpu'))
        
        packed_embedded = nn.utils.rnn.pack_padded_sequence(bert_sentence, sorted_seq_lengths.to('cpu'),
                                                            batch_first=self.batch_first)
        if self.rnn_type in ['rnn', 'gru']:
            packed_output, hidden = self.rnn(packed_embedded)
        else:
            packed_output, (hidden, cell) = self.rnn(packed_embedded)

        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=self.batch_first)
        output = output[desorted_indices]
        hidden = hidden[desorted_indices]

        batch_size, max_seq_len, hidden_dim = output.shape
        hidden = torch.mean(torch.reshape(hidden, [batch_size, -1, hidden_dim]), dim=1)
        output = torch.sum(output, dim=1)
        fc_input = self.dropout(output + hidden)
        out = self.fc_rnn(fc_input)

        return out,fc_input


class TransformersRCNN(BaseModel):
    def __init__(self, transformer_model, cache_dir, force_download,rnn_type, hidden_dim, n_layers, bidirectional,batch_first,
                 dropout, class_num,is_train):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.transformer_config = AutoConfig.from_pretrained(transformer_model, cache_dir=cache_dir,
                                                             force_download=force_download)
        self.transformer_model = AutoModel.from_pretrained(transformer_model, config=self.transformer_config,
                                                           cache_dir=cache_dir, force_download=force_download)
        for name, param in self.transformer_model.named_parameters():
            param.requires_grad = is_train

        # (*** 关键修正 4 ***)
        hidden_size = self.transformer_model.config.hidden_size
        
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(hidden_size,
                               hidden_dim,
                               num_layers=n_layers,
                               bidirectional=bidirectional,
                               batch_first=batch_first,
                               dropout=dropout)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(hidden_size,
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)
        else:
            self.rnn = nn.RNN(hidden_size,
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)

        self.fc = nn.Linear(hidden_dim * n_layers, class_num)
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

    def forward(self, input_ids, attention_masks, text_lengths):
        # (适配 distilbert)
        transformer_output = self.transformer_model(input_ids, attention_mask=attention_masks)
        sentence_out = transformer_output[0]
        # (适配 distilbert: [CLS] token is at [:, 0, :])
        cls = sentence_out[:, 0, :].unsqueeze(1).repeat(1, sentence_out.shape[1], 1)
        
        sentence_out = sentence_out + cls
        
        bert_sentence, sorted_seq_lengths, desorted_indices = prepare_pack_padded_sequence(sentence_out, text_lengths.to('cpu'))
        
        packed_embedded = nn.utils.rnn.pack_padded_sequence(bert_sentence, sorted_seq_lengths.to('cpu'),
                                                            batch_first=self.batch_first)
        if self.rnn_type in ['rnn', 'gru']:
            packed_output, hidden = self.rnn(packed_embedded)
        else:
            packed_output, (hidden, cell) = self.rnn(packed_embedded)

        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=self.batch_first)
        output = output[desorted_indices]

        batch_size, max_seq_len, hidden_dim = output.shape
        out = torch.transpose(output.relu(), 1, 2)
        out_embedding = F.max_pool1d(out, int(max_seq_len)).squeeze()
        out = self.fc(out_embedding)

        return out,out_embedding