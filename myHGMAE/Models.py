from typing import List
import torch
import torch.nn as nn
from functools import partial
import dgl
from dgl.ops import edge_softmax
import dgl.function as fn
from dgl.utils import expand_as_pair


# from openhgnn.layers.MetapathConv import MetapathConv
# from openhgnn.utils import extract_metapaths
# from openhgnn.layers.macro_layer.SemanticConv import SemanticAttention

class NormLayer(nn.Module):
    def __init__(self, hidden_dim, norm_type):
        super().__init__()
        if norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(hidden_dim)
        elif norm_type == "graphnorm":
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))

            self.mean_scale = nn.Parameter(torch.ones(hidden_dim))
        else:
            raise NotImplementedError

    def forward(self, graph, x):
        tensor = x
        if self.norm is not None and type(self.norm) != str:
            return self.norm(tensor)
        elif self.norm is None:
            return tensor

        batch_list = graph.batch_num_nodes
        batch_size = len(batch_list)
        batch_list = torch.Tensor(batch_list).long().to(tensor.device)
        batch_index = torch.arange(batch_size).to(tensor.device).repeat_interleave(batch_list)
        batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor)
        mean = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        mean = mean.scatter_add_(0, batch_index, tensor)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)

        sub = tensor - mean * self.mean_scale

        std = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)
        return self.weight * sub / std + self.bias


def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    elif name == "graphnorm":
        return partial(NormLayer, norm_type="groupnorm")
    else:
        return None


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1) beta : att_mp
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)
        out_emb = (beta * z).sum(1)  # (N, D * K)
        att_mp = beta.mean(0).squeeze()
        # out_emb:
        return out_emb, att_mp


class GATConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=True,
                 norm=None,
                 concat_out=True):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree

        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(size=(num_heads * out_feats,)))
        else:
            self.register_buffer('bias', None)
        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = nn.Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

        self.concat_out = concat_out  # 改动：相比于dgl，新增
        self.norm = norm  # 改动：增
        if norm is not None:
            self.norm = norm(num_heads * out_feats)

    def reset_parameters(self):
        """

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
            # 改动：相比于dgl，删了
            # if self.res_fc.bias is not None:
            #     nn.init.constant_(self.res_fc.bias, 0)

    def set_allow_zero_in_degree(self, set_value):
        r"""
        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
          The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, get_attention=False):
        """
            feat: Tensor of shape [num_nodes,feat_dim]
            改：相比于dgl，没有edge_weight
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise RuntimeError('There are 0-in-degree nodes in the graph, '
                                       'output for those nodes will be invalid. '
                                       'This is harmful for some applications, '
                                       'causing silent performance regression. '
                                       'Adding self-loop on the input graph by '
                                       'calling `g = dgl.add_self_loop(g)` will resolve '
                                       'the issue. Setting ``allow_zero_in_degree`` '
                                       'to be `True` when constructing this module will '
                                       'suppress the check and let the code run.')

            if isinstance(feat, tuple):
                src_prefix_shape = feat[0].shape[:-1]
                dst_prefix_shape = feat[1].shape[:-1]
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    feat_src = self.fc(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats)
                    feat_dst = self.fc(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats)
                else:
                    feat_src = self.fc_src(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats)
                    feat_dst = self.fc_dst(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats)
            else:
                src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    *src_prefix_shape, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
                    h_dst = h_dst[:graph.number_of_dst_nodes()]
                    dst_prefix_shape = (graph.number_of_dst_nodes(),) + dst_prefix_shape[1:]
            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))
            # e[e == 0] = -1e3
            # e = graph.edata.pop('e')
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            # 改：没有edge_weight
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']

            # bias
            if self.bias is not None:
                rst = rst + self.bias.view(
                    *((1,) * len(dst_prefix_shape)), self._num_heads, self._out_feats)

            # residual
            if self.res_fc is not None:
                # Use -1 rather than self._num_heads to handle broadcasting
                resval = self.res_fc(h_dst).view(*dst_prefix_shape, -1, self._out_feats)
                rst = rst + resval

            # 改：增加了concat_out，可外提
            if self.concat_out:
                rst = rst.flatten(1)
            else:
                rst = torch.mean(rst, dim=1)
            # 改：增加了norm，可外提
            if self.norm is not None:
                rst = self.norm(rst)

            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst


class HANLayer(nn.Module):
    """
    HAN layer.

    Arguments
    ---------
    meta_paths : int
        Number of metapaths
    in_dim : int
        input feature dimension
    out_dim : int
        output feature dimension
    layer_num_heads : number of attention heads (each GATConv)
    dropout : Dropout probability

    Inputs
    ------
    g : DGLHeteroGraph
        The heterogeneous graph
    h : tensor
        Input features

    Outputs
    -------
    tensor
        The output feature
    """

    def __init__(self, num_metapath, in_dim, out_dim, layer_num_heads,
                 feat_drop, attn_drop, negative_slope, residual, activation, norm, concat_out):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()  # openhgnn里用的是nn.ModuleDict
        for i in range(num_metapath):
            self.gat_layers.append(GATConv(
                in_dim, out_dim, layer_num_heads,
                feat_drop, attn_drop, negative_slope, residual, activation, norm=norm, concat_out=concat_out))
        self.semantic_attention = SemanticAttention(in_size=out_dim * layer_num_heads)  # macro

    def forward(self, gs, h):
        semantic_embeddings = []
        for i, new_g in enumerate(gs):
            semantic_embeddings.append(self.gat_layers[i](new_g, h).flatten(1))  # flatten because of att heads
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # (N, M, D * K)
        out, att_mp = self.semantic_attention(semantic_embeddings)  # (N, D * K)

        return out, att_mp


class HAN(nn.Module):
    '''
    HAN : contains several HANLayers
        when num_layers=1, layer(in_dim,out_dim,num_out_heads)
        when num_layers>1:
            layer_1(in_dim,hidden_dim,num_heads),
            layer_2(hidden_dim*num_heads,hidden_dim,num_heads)
            ...
            layers_n(hidden_dim*num_heads,out_dim,num_out_heads)

    HANLayers : contains several GATConv(in_dim,out_dim), the number of GATLayers is up to num_metapaths

    Parameters
    ------------
    num_metapath : int
        Number of metapaths.
    in_dim : int
        Input feature dimension.
    hidden_dim : int
        Hidden layer dimension.
    out_dim : int
        Output feature dimension.
    num_heads : int
        Number of attentions heads (Multiple HANLayers all use the same num_heads)
    num_out_heads : int
        Number of attentions heads of output projection
    dropout : float
        Dropout probability.
    encoding : bool
        True means encoder, False means decoder
    """

    '''

    def __init__(self,
                 num_metapath,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 num_layers,
                 num_heads,
                 num_out_heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 norm,
                 concat_out=False,
                 encoding=False
                 ):
        super(HAN, self).__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.han_layers = nn.ModuleList()
        self.concat_out = concat_out

        self.activation = create_activation(activation)
        last_activation = create_activation(activation) if encoding else create_activation(None)

        last_residual = (encoding and residual)
        last_norm = norm if encoding else None

        if num_layers == 1:
            self.han_layers.append(HANLayer(num_metapath,
                                            in_dim, out_dim, num_out_heads,
                                            feat_drop, attn_drop, negative_slope, last_residual, last_activation,
                                            norm=last_norm, concat_out=concat_out))
        else:
            # input projection (no residual)
            self.han_layers.append(HANLayer(num_metapath,
                                            in_dim, hidden_dim, num_heads,
                                            feat_drop, attn_drop, negative_slope, residual, self.activation, norm=norm,
                                            concat_out=concat_out))
            # hidden layers
            for l in range(1, num_layers - 1):
                # due to multi-head, the in_dim = hidden_dim * num_heads
                self.han_layers.append(HANLayer(num_metapath,
                                                hidden_dim * num_heads, hidden_dim, num_heads,
                                                feat_drop, attn_drop, negative_slope, residual, self.activation,
                                                norm=norm, concat_out=concat_out))
            # output projection
            self.han_layers.append(HANLayer(num_metapath,
                                            hidden_dim * num_heads, out_dim, num_out_heads,
                                            feat_drop, attn_drop, negative_slope, last_residual,
                                            activation=last_activation, norm=last_norm, concat_out=concat_out))

    def forward(self, gs: List[dgl.DGLGraph], h: torch.Tensor):
        # gs is masked metapath_reachable_graph
        for han_layer in self.han_layers:
            h, att_mp = han_layer(gs, h)
        return h, att_mp
        # 用openhgnn实现时， att_mp或许可以通过HAN.mod_dict[].get_emb


