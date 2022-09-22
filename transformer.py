#!/usr/bin/env python3
# Author: Domenick Mifsud
#───────#
import math
import copy
#────#
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList, Module
from torch.nn.parameter import Parameter
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear as NDQL
from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer
#────#
from utils import get_norm
'''───────────────────────────── transformer.py ─────────────────────────────'''
# This file contains the NDT-U model.

class MHA(Module):
    def __init__(self, config, n_neurons):
        super(MHA, self).__init__()
        self.config = config

        # If using undivided attention, each head needs 'input_dim' dimensions
        self.packed_dim_size = n_neurons * config['model']['n_heads'] if (
            config['model']['undivided_attn']
        ) else n_neurons

        # MHA uses a packed tensor, Queries Keys and Values all share the same weight matrix
        self.in_proj_weight = Parameter(torch.empty((3 * self.packed_dim_size, n_neurons)))
        self.in_proj_bias = Parameter(torch.empty(3 * self.packed_dim_size))
        self.out_proj = NDQL(self.packed_dim_size, n_neurons)

        # Init QKV weights and all biases
        xavier_uniform_(self.in_proj_weight)
        constant_(self.in_proj_bias, 0.)
        constant_(self.out_proj.bias, 0.)

    def forward(self, src, attn_mask=None):
        # Use the same weight matrix then seperate 
        self.q, self.k, self.v = torch._C._nn.linear(src, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)

        # If using undivided attention the view shape is [T x (B * n_heads) x N]
        self.view_shape = (src.shape[0], src.shape[1] * self.config['model']['n_heads'], src.shape[2]) if (
            self.config['model']['undivided_attn']
        ) else (
            # If using standard MHA the view shape is [T x (B * n_heads) x (N // n_heads)]
            (src.shape[0], src.shape[1] * self.config['model']['n_heads'], src.shape[2] // self.config['model']['n_heads'])
        )
        self.q = self.q.contiguous().view(*self.view_shape).transpose(0, 1) / math.sqrt(self.view_shape[2])
        self.k = self.k.contiguous().view(*self.view_shape).transpose(0, 1)
        self.v = self.v.contiguous().view(*self.view_shape).transpose(0, 1)

        # Create the attention matrix [T x T]
        self.attn = torch.bmm(self.q, self.k.transpose(-2, -1))

        # Restrict how far in past / future each timestep can attend to
        if attn_mask is not None:
            self.attn += attn_mask

        # Apply softmax and dropout to attention matrix    
        self.attn = F.softmax(self.attn, dim=-1)
        if self.training:
            self.attn = F.dropout(self.attn, p=self.config['model']['dropout_attention'] )
        
        # Multiply attention matrix (QK) and values (V)
        self.attn_output = torch.bmm(self.attn, self.v).transpose(0, 1)
        self.attn_output = self.attn_output.contiguous().view(src.shape[0] * src.shape[1], self.packed_dim_size)

        # Project to proper size ([T x B x N]) and return
        return torch._C._nn.linear(self.attn_output, self.out_proj.weight, self.out_proj.bias).view(*src.shape)

class EncoderLayer(TransformerEncoderLayer):
    '''A simplified version of pytorch's TransformerEncoderLayer.

     Attributes:
        norm1 (function): Normalization specified in config.
        norm2 (function): Normalization specified in config.
        dropout (nn.Dropout): Dropout.
        dropout1 (nn.Dropout): Dropout.
        dropout2 (nn.Dropout): Dropout.
    '''
    def __init__(self, config, n_neurons, seq_len):
        ''' init EncoderLayer

        Args:
            config (dict): The config.
            n_neurons (int): Number of neurons.
            seq_len (int): Length of trial + forward pass
        '''
        super().__init__(
            n_neurons, nhead=1,
            dim_feedforward=config['model']['hidden_size'],
            dropout=config['model']['dropout'],
            activation=config['model']['activation']
        )
        self.self_attn = MHA(config, n_neurons)
        # self.self_attn = UndividedMultiheadAttention(config, n_neurons)

        self.norm1 = get_norm(config, n_neurons)
        self.norm2 = get_norm(config, n_neurons)

        self.dropout = nn.Dropout(config['model']['dropout'])
        self.dropout1 = nn.Dropout(config['model']['dropout'])
        self.dropout2 = nn.Dropout(config['model']['dropout'])

        # http://www.cs.toronto.edu/~mvolkovs/ICML2020_tfixup.pdf
        temp_state_dic = {}
        n_layers = config['model']['n_layers']
        for name, param in self.named_parameters():
            if name in ["linear1.weight", "linear2.weight", "self_attn.out_proj.weight"]:
                temp_state_dic[name] = param * (0.67 * (n_layers) ** (- 1. / 4.))
        for name in self.state_dict():
            if name not in temp_state_dic:
                temp_state_dic[name] = self.state_dict()[name]
        self.load_state_dict(temp_state_dic)

    def forward(self, src, attn_mask=None):
        '''Forward pass.

        Args:
            src (Tensor): A batch of data. Size=[T, B, N]
            attn_mask (Tensor, Optional): How far each timestep can attend to.
                                                  Size=[T, T]
        '''
        residual = src
        src = self.norm1(src)
        src2 = self.self_attn(src, attn_mask)
        src= residual + self.dropout1(src2)

        residual = src
        src = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src2)

        return src


class Encoder(Module):
    '''A simplified version of pytorch's TransformerEncoder.

     Attributes:
        layers (ModuleList): A list of EncoderLayers.
        norm (function): Normalization specified in config.
    '''
    def __init__(self, config, n_neurons, encoder_layer, seq_len):
        ''' init Encoder.

        Args:
            config (dict): The config.
            n_neurons (int): Number of neurons.
            encoder_layer (EncoderLayer): The encoder to stack.
            seq_len (int): Length of trial + forward pass
        '''
        super().__init__()
        n_layers = config['model']['n_layers']
        self.layers = ModuleList([copy.deepcopy(encoder_layer) for i in range(n_layers)])
        self.norm = get_norm(config, n_neurons)

    def forward(self, src, attn_mask=None):
        ''' PlaceHolder
        '''
        for layer in self.layers: # send through each EncoderLayer's forward()
            src = layer(src, attn_mask)
        if self.norm is not None: # final norm
            src = self.norm(src)
        return src

class Transformer(Module):
    '''Simplified version of Joel Ye's NDT with unvdivided attention added.
    (https://github.com/snel-repo/neural-data-transformers)
    (https://www.biorxiv.org/content/10.1101/2021.01.16.426955v3.full.pdf)

     Attributes:
        config (dict): The config.
        name (str): The model name.
        n_heldin (int): Number of held-in neurons.
        n_neurons (int): Number of neurons.
        tr_length (int): Trial length (no forward).
        seq_len (int): Trial length + forward pass length.
        scale (float): What to scale neruon activity by.
        n_layers (int): Number of layers.
        rate_dropout (float): The percentage of dropout applied to the rates.
        embedding_dropout (float): The percentage of dropout applied after
                                   adding the embeddings.
        encoder (Encoder): The encoder, which is stacked EndocerLayers.
        decoder (nn.Linear): The decoder, which is Linear layers with an
                             activiation in between.
        attn_mask (Tensor): How far each timestep can attend to. Size=[T, T]
        loss_prob_mask (Tensor): Cached loss probability tensor.
        zero_prob_mask (Tensor): Cached zero mask probability tensor.
        random_prob_mask (Tensor): Cached random probability tensor.
        classifier (function): nn.PoissonNLLLoss().
    '''

    def __init__(self, config, dataset, name):
        """init Transformer

        Args:
            config (dict): The config.
            dataset (Dataset): The Dataset used to create the DataLoaders.
            name (str): The model name.
        """
        super().__init__()
        self.config = config
        self.name = name

        self.n_heldin = dataset.n_heldin
        self.n_neurons = dataset.n_neurons
        self.seq_len = config['train']['seq_len']
        self.max_train_spks = dataset.max_train_spks

        self.scale = math.sqrt(self.n_neurons)
        self.n_layers = config['model']['n_layers']
        self.rate_dropout = nn.Dropout(config['model']['dropout_rates'])
        self.embedding_dropout = nn.Dropout(p=config['model']['dropout_embedding'])

        pe = torch.zeros(self.seq_len, self.n_neurons)
        position = torch.arange(0, self.seq_len, dtype=torch.float).unsqueeze(1)
        self.register_buffer('pe', position.long())
        self.pos_embedding = nn.Embedding(self.seq_len, self.n_neurons)

        encoder_layer = EncoderLayer(config, self.n_neurons, self.seq_len)
        self.encoder = Encoder(config, self.n_neurons, encoder_layer, self.seq_len)
        self.decoder = nn.Linear(self.n_neurons, dataset.n_neurons)

        self.attn_mask = None
        self.loss_prob_mask = None
        self.zero_prob_mask = None
        self.random_prob_mask = None

        self.classifier = nn.PoissonNLLLoss(reduction='none')

        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(
            -config['model']['initrange'], config['model']['initrange'])

        if config['model']['normal_init']:
            std = (2 / (5 * (self.n_neurons))) ** 0.5
            for parameter in self.parameters():
                if parameter.dim() > 1:
                    nn.init.normal_(parameter, mean=0.0, std=std)

    def forward(self, spikes, labels=None):
        ''' Forward pass.

        Args:
            spikes (Tensor): A batch of spiking data. Size=[B x T x N]
            labels (Tensor): A labels of spiking data. Size=[B x T x N]
        Returns:
            final_loss (Tensor): The loss. Size=[1]
            pred_rates (Tensor): The predicted rates. Size=[B, T, N]
        '''
        if not self.training: 
            spikes = torch.clamp(spikes, max=self.max_train_spks)

        spikes = spikes.permute(1, 0, 2) * self.scale # [B x T x N] -> [T x B x N]
        spikes += self.pos_embedding(self.pe)
        spikes = self.embedding_dropout(spikes)

        attn_mask = self.get_attn_mask(spikes) # [T, T]
        output = self.encoder(spikes, attn_mask)
        output = self.rate_dropout(output)

        pred_rates = self.decoder(output).permute(1, 0, 2)  # [T x B x N] ->  [B x T x N]
        if labels == None: return pred_rates

        loss = self.classifier(pred_rates, labels)

        # lt_loss = loss[:, -1, :].mean()
        # masked_loss = loss[labels != -100].mean()

        return loss, pred_rates

    def get_attn_mask(self, src):
        ''' Gets attention mask stored on memory or creates a new one.

        Args:
            src (Tensor): A batch of spiking data. Size=[B x T x N]
        Returns:
            mask (Tensor): The attention mask. Size=[T, T]
        '''
        if self.attn_mask != None: # Use cached version if already created.
            return self.attn_mask
        seq_len = src.size(0)
        context_forward = self.config['model']['context_forward']
        context_backward = self.config['model']['context_backward']
        ones = torch.ones(seq_len, seq_len, device=src.device)
        forw_mask = (torch.triu(ones, diagonal=-context_forward) == 1).transpose(0, 1)
        back_mask = (torch.triu(ones, diagonal=-context_backward) == 1)
        mask = (forw_mask & back_mask).float()
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        self.attn_mask = mask
        return mask

    def preprocess_batch(self, batch, expand_p, heldout_spikes):
        ''' Zero masks and randomizes the batch. Also returns the labels of which indicies
        should be used to compute the loss with.

        Args:
            batch (Tensor): A batch of spiking data. Size=[B x T x N]
            expand_p (float): probability to expand the mask across timesteps.
            heldout_spikes (Tensor): The held-out spikes. Size=[B x T x N]
            heldin_spikes (Tensor): The held-in spikes. Size=[B x T x N]
        Returns:
            batch (Tensor): The batch of spiking data, masked and randomized. Size=[B x T x N]
            labels (Tensor): The labels of spiking data, indicating which indicies
                             should be used to compute the loss with. Size=[B x T x N]
        '''
        batch = batch.clone() # make sure we don't corrupt the input data (which is stored in memory)
        labels = batch.clone()
        config = self.config

        # Expand if expand probability is greater than the number generated.
        should_expand = config['train']['mask_max_span'] > 1 and expand_p > 0.0 and torch.rand(1).item() < expand_p
        width =  torch.randint(1, config['train']['mask_max_span'] + 1, (1, )).item() if should_expand else 1
        loss_ratio = config['model']['loss_ratio'] if width == 1 else config['model']['loss_ratio'] / width

        # Which indicies shold the loss be computed with
        if self.loss_prob_mask is None or self.loss_prob_mask.size() != labels.size():
            timestep_shape = labels[:, :, 0].shape # N x T
            self.loss_prob_mask = torch.full(timestep_shape, loss_ratio, device=batch.device, dtype=torch.float32)
        loss_mask = torch.bernoulli(self.loss_prob_mask)

        # Expand
        if width > 1:
            kernel = torch.ones(width, device=loss_mask.device).view(1, 1, -1)
            expanded_mask = F.conv1d(loss_mask.unsqueeze(1), kernel, padding= width// 2).clamp_(0, 1)
            if width % 2 == 0:
                expanded_mask = expanded_mask[...,:-1] # crop if even (we've added too much padding)
            loss_mask = expanded_mask.squeeze(1)

        loss_mask = loss_mask.bool().unsqueeze(2).expand_as(labels)
        loss_mask[:, -1, :] = True
        labels[~loss_mask] = -100

        # Zero mask
        if self.zero_prob_mask is None or self.zero_prob_mask.size() != labels.size():
            zero_mask_ratio = config['model']['mask_ratio']
            self.zero_prob_mask = torch.full(labels.shape, zero_mask_ratio, device=batch.device, dtype=torch.float32)
        # indices_zero_masked = torch.bernoulli(self.zero_prob_mask).bool()
        indices_zero_masked = torch.bernoulli(self.zero_prob_mask).bool() & loss_mask
        batch[indices_zero_masked] = 0

        # Randomize
        if self.random_prob_mask is None or self.random_prob_mask.size() != labels.size():
            randomize_ratio = config['model']['random_ratio']
            self.random_prob_mask = torch.full(labels.shape, randomize_ratio, device=batch.device, dtype=torch.float32)
        # indices_randomized = torch.bernoulli(self.random_prob_mask).bool() & ~indices_zero_masked
        indices_randomized = torch.bernoulli(self.random_prob_mask).bool() & loss_mask & ~indices_zero_masked
        random_spikes = torch.randint(batch.max().long(), labels.shape, dtype=torch.long, device=batch.device)
        batch[indices_randomized] = random_spikes.float()[indices_randomized]

        # Add fake heldout and forward
        batch = torch.cat([batch, torch.zeros_like(heldout_spikes)], -1)
        labels = torch.cat([labels, heldout_spikes], -1)

        return batch, labels
