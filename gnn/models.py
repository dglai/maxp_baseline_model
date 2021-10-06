#-*- coding:utf-8 -*-

# Author:james Zhang

"""
    Three common GNN models.
"""

import torch.nn as thnn
import torch.nn.functional as F
import dgl.nn as dglnn


class GraphSageModel(thnn.Module):

    def __init__(self,
                 in_feats,
                 hidden_dim,
                 n_layers,
                 n_classes,
                 activation=F.relu,
                 dropout=0):
        super(GraphSageModel, self).__init__()
        self.in_feats = in_feats
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.activation = activation
        self.dropout = thnn.Dropout(dropout)

        self.layers = thnn.ModuleList()

        # build multiple layers
        self.layers.append(dglnn.SAGEConv(in_feats=self.in_feats,
                                          out_feats=self.hidden_dim,
                                          aggregator_type='mean'))
                                          # aggregator_type = 'pool'))
        for l in range(1, (self.n_layers - 1)):
            self.layers.append(dglnn.SAGEConv(in_feats=self.hidden_dim,
                                              out_feats=self.hidden_dim,
                                              aggregator_type='mean'))
                                              # aggregator_type='pool'))
        self.layers.append(dglnn.SAGEConv(in_feats=self.hidden_dim,
                                          out_feats=self.n_classes,
                                          aggregator_type='mean'))
                                          # aggregator_type = 'pool'))

    def forward(self, blocks, features):
        h = features

        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)

        return h


class GraphConvModel(thnn.Module):

    def __init__(self,
                 in_feats,
                 hidden_dim,
                 n_layers,
                 n_classes,
                 norm,
                 activation,
                 dropout):
        super(GraphConvModel, self).__init__()
        self.in_feats = in_feats
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.norm = norm
        self.activation = activation
        self.dropout = thnn.Dropout(dropout)

        self.layers = thnn.ModuleList()

        # build multiple layers
        self.layers.append(dglnn.GraphConv(in_feats=self.in_feats,
                                           out_feats=self.hidden_dim,
                                           norm=self.norm,
                                           activation=self.activation,))
        for l in range(1, (self.n_layers - 1)):
            self.layers.append(dglnn.GraphConv(in_feats=self.hidden_dim,
                                               out_feats=self.hidden_dim,
                                               norm=self.norm,
                                               activation=self.activation))
        self.layers.append(dglnn.GraphConv(in_feats=self.hidden_dim,
                                           out_feats=self.n_classes,
                                           norm=self.norm,
                                           activation=self.activation))

    def forward(self, blocks, features):
        h = features

        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.dropout(h)

        return h


class GraphAttnModel(thnn.Module):

    def __init__(self,
                 in_feats,
                 hidden_dim,
                 n_layers,
                 n_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop
                 ):
        super(GraphAttnModel, self).__init__()
        self.in_feats = in_feats
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.heads = heads
        self.feat_dropout = feat_drop
        self.attn_dropout = attn_drop
        self.activation = activation

        self.layers = thnn.ModuleList()

        # build multiple layers
        self.layers.append(dglnn.GATConv(in_feats=self.in_feats,
                                         out_feats=self.hidden_dim,
                                         num_heads=self.heads[0],
                                         feat_drop=self.feat_dropout,
                                         attn_drop=self.attn_dropout,
                                         activation=self.activation))

        for l in range(1, (self.n_layers - 1)):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.layers.append(dglnn.GATConv(in_feats=self.hidden_dim * self.heads[l - 1],
                                             out_feats=self.hidden_dim,
                                             num_heads=self.heads[l],
                                             feat_drop=self.feat_dropout,
                                             attn_drop=self.attn_dropout,
                                             activation=self.activation))

        self.layers.append(dglnn.GATConv(in_feats=self.hidden_dim * self.heads[-2],
                                         out_feats=self.n_classes,
                                         num_heads=self.heads[-1],
                                         feat_drop=self.feat_dropout,
                                         attn_drop=self.attn_dropout,
                                         activation=None))

    def forward(self, blocks, features):
        h = features

        for l in range(self.n_layers - 1):
            h = self.layers[l](blocks[l], h).flatten(1)

        logits = self.layers[-1](blocks[-1],h).mean(1)

        return logits

