import torch
import torch.nn as nn
from openhgnn.models import BaseModel
import dgl
import Models
from dgl import DropEdge
from functools import partial
import torch.nn.functional as F


def sce_loss(x, y, gamma=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (1 - (x * y).sum(dim=-1)).pow_(gamma)

    loss = loss.mean()
    return loss


def create_loss_func(loss_func, gamma):
    if loss_func == "mse":
        criterion = nn.MSELoss()
    elif loss_func == "sce":
        criterion = partial(sce_loss, gamma=gamma)
    else:
        raise NotImplementedError
    return criterion


def get_mask_rate(input_mask_rate, get_min=False, epoch=None):
    try:
        return float(input_mask_rate)
    except ValueError:
        if "," in input_mask_rate:  # 0.6,-0.1,0.4 stepwise increment/decrement
            mask_rate = [float(i) for i in input_mask_rate.split(',')]
            assert len(
                mask_rate) == 3, "input_mask_rate should be a float number (0-1), or in the format of 'min,delta," \
                                 "max', '0.6,-0.1,0.4', for example "
            start = mask_rate[0]
            step = mask_rate[1]
            end = mask_rate[2]
            if get_min:
                return min(start, end)
            else:
                cur_mask_rate = start + epoch * step
                if cur_mask_rate < min(start, end) or cur_mask_rate > max(start, end):
                    return end
                return cur_mask_rate
        else:
            raise NotImplementedError(
                "input_mask_rate should be a float number (0-1), or in the format of 'min,delta,max', '0.6,-0.1,0.4', "
                "for example")


class HGMAE(BaseModel):
    r'''

    Parameter
    ----------
    metapaths_dict: dict[str, list[etype]]
        Dict from meta path name to meta path.
    category : string
        The category of the nodes to be classificated.
    in_dim : int
        Dim of input feats
    hidden_dim : int
        Dim of encoded embedding.
    num_layers : int
        Number of layers of HAN encoder and decoder.
    num_heads : int
        Number of attention heads of hidden layers in HAN encoder and decoder.
    num_out_heads : int
        Number of attention heads of output projection in HAN encoder and decoder.
    feat_drop : float, optional
        Dropout rate on feature.
    attn_drop : float, optional
        Dropout rate on attention weight.
    negative_slope : float, optional
        LeakyReLU angle of negative slope. Defaults: ``0.2``.
    residual : bool, optional
        If True, use residual connection. Defaults: ``False``.
    activation : str or None
        Activation function, str in {'relu','gelu','prelu','elu'} or None. Defaults: ``'prelu'``
    norm : str or None
        NormLayers, str in {'batchnorm','layernorm','graphnorm'} or None. Defaults: ``None``
    concat_out : bool
        Concat outputs of multi-heads GATConv. Defaults: ``True``
    '''

    @classmethod
    def build_model_from_args(cls, args, metapaths_dict: dict):
        norm = Models.create_norm(args.norm)
        return cls(
            metapaths_dict=metapaths_dict,
            category=args.category,
            in_dim=args.in_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            num_out_heads=args.num_out_heads,
            activation=args.activation,
            feat_drop=args.feat_drop,
            attn_drop=args.attn_drop,
            negative_slope=args.negative_slope,
            residual=args.residual,
            norm=norm,
            concat_out=True,
            loss_func=args.loss_func,

            # Metapath-based Edge Reconstruction
            use_mp_edge_recon=args.use_mp_edge_recon,
            mp_edge_recon_loss_weight=args.mp_edge_recon_loss_weight,
            mp_edge_mask_rate=args.mp_edge_mask_rate,
            mp_edge_gamma=args.mp_edge_gamma,

            # Type-specific Attribute Restoration
            attr_restore_gamma=args.attr_restore_gamma,
            attr_restore_loss_weight=args.attr_restore_loss_weight,
            node_mask_rate=args.node_mask_rate,
            feat_replace_rate=args.feat_replace_rate,
            feat_unchanged_rate=args.feat_unchanged_rate,

        )

    def __init__(self, metapaths_dict, category,
                 in_dim, hidden_dim, num_layers, num_heads, num_out_heads,
                 feat_drop, attn_drop, negative_slope=0.2, residual=False,
                 activation='prelu', norm=None, concat_out=True, loss_func='sce',
                 use_mp_edge_recon=True, mp_edge_recon_loss_weight=1, mp_edge_mask_rate=0.7, mp_edge_gamma=3,
                 attr_restore_loss_weight=1, attr_restore_gamma=1, node_mask_rate='0.5,0.005,0.8',
                 feat_replace_rate=0.3, feat_unchanged_rate=0.2

                 ):

        super(HGMAE, self).__init__()
        self.metapaths_dict = metapaths_dict
        self.num_metapaths = len(metapaths_dict)
        self.category = category
        self.in_dim = in_dim  # original feat dim
        self.hidden_dim = hidden_dim  # emb dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_out_heads = num_out_heads
        self.activation = activation
        self.feat_drop = feat_drop
        self.attn_drop = attn_drop
        self.negative_slope = negative_slope
        self.residual = residual
        self.norm = norm
        self.concat_out = concat_out
        self.loss_func = loss_func

        assert self.hidden_dim % self.num_heads == 0
        assert self.hidden_dim % self.num_out_heads == 0

        # The input dimensions of the encoder and decoder are the same
        self.enc_dec_input_dim = self.in_dim

        # num head: encoder
        enc_hidden_dim = self.hidden_dim // self.num_heads
        enc_num_heads = self.num_heads

        # num head: decoder
        dec_hidden_dim = self.hidden_dim // self.num_out_heads
        dec_num_heads = self.num_out_heads

        dec_in_dim = self.hidden_dim

        # NOTE:
        # hidden_dim of HAN and hidden_dim of HGMAE are different,
        # the former one is the hidden_dim insides the HAN model,
        # the latter one is actually the dim of embeddings produced by the encoder insides the HGMAE,
        # The parameter hidden_dim refers specifically to the embedding produced by HGMAE encoder

        # encoder
        # when concat_out=True,
        # actual output dim of encoder = out_dim * num_out_heads
        #                              = enc_hidden_dim * enc_num_heads
        #                              = hidden_dim (param, that is dim of emb)
        #                              = dec_in_dim
        # emb_dim of encoder = self.hidden_dim (param)
        self.encoder = Models.HAN(
            num_metapaths=self.num_metapaths,
            in_dim=self.in_dim,
            hidden_dim=enc_hidden_dim,
            out_dim=enc_hidden_dim,
            num_layers=self.num_layers,
            num_heads=enc_num_heads,
            num_out_heads=enc_num_heads,
            activation=self.activation,
            feat_drop=self.feat_drop,
            attn_drop=self.attn_drop,
            negative_slope=self.negative_slope,
            residual=self.residual,
            norm=self.norm,
            concat_out=self.concat_out,
            encoding=True
        )

        # decoder
        self.decoder = Models.HAN(
            num_metapaths=self.num_metapaths,
            in_dim=dec_in_dim,
            hidden_dim=dec_hidden_dim,
            out_dim=self.enc_dec_input_dim,
            num_layers=1,
            num_heads=dec_num_heads,
            num_out_heads=dec_num_heads,
            activation=self.activation,
            feat_drop=self.feat_drop,
            attn_drop=self.attn_drop,
            negative_slope=self.negative_slope,
            residual=self.residual,
            norm=self.norm,
            concat_out=self.concat_out,
            encoding=False
        )

        self.__cached_gs = None  # cached metapath reachable graphs
        self.__cached_mps = None  # cached metapath adjacency matrices (SparseMatrix)

        # Metapath-based Edge Reconstruction
        self.use_mp_edge_recon = use_mp_edge_recon
        self.mp_edge_recon_loss_weight = mp_edge_recon_loss_weight
        self.mp_edge_mask_rate = mp_edge_mask_rate
        self.mp_edge_gamma = mp_edge_gamma
        self.mp_edge_recon_loss = create_loss_func(self.loss_func, self.mp_edge_gamma)
        self.encoder_to_decoder_edge_recon = nn.Linear(dec_in_dim, dec_in_dim, bias=False)

        # Type-specific Attribute Restoration
        self.attr_restore_gamma = attr_restore_gamma
        self.enc_mask_token = nn.Parameter(torch.zeros(1, self.enc_dec_input_dim)) # learnable mask token [M]
        self.encoder_to_decoder_attr_restore = nn.Linear(dec_in_dim, dec_in_dim, bias=False)
        self.attr_restora_loss=create_loss_func(self.loss_func,self.attr_restore_gamma)
        self.attr_restore_loss_weight = attr_restore_loss_weight
        self.node_mask_rate = node_mask_rate
        self.feat_unchanged_rate = feat_unchanged_rate
        self.feat_replace_rate = feat_replace_rate

        assert self.feat_replace_rate + self.feat_unchanged_rate < 1, "feat_replace_rate + feat_unchanged_rate must " \
                                                                      "be smaller than 1 "

    # mps: a list of metapath-based adjacency matrices (SparseMatrix)
    def get_mps(self, hg: dgl.DGLHeteroGraph):
        if self.__cached_mps is None:
            self.__cached_mps = [dgl.metapath_reachable_graph(hg, mp).adjacency_matrix() for mp in
                                 self.metapaths_dict.values()]
        return self.__cached_mps

    # gs: a list of meta path reachable graphs that only contain topological structures
    # without edge and node features
    def mps_to_gs(self, mps):
        if self.__cached_gs is None:
            gs = []
            for mp in mps:
                indices = mp.indices()
                cur_graph = dgl.graph((indices[0], indices[1]))
                gs.append(cur_graph)
                self.__cached_gs = gs

            return gs
        else:
            return self.__cached_gs

    def mask_mp_edge_reconstruction(self, mps, feat, epoch):
        masked_gs = self.mps_to_gs(mps)
        cur_mp_edge_mask_rate = get_mask_rate(self.mp_edge_mask_rate, epoch=epoch)
        drop_edge = DropEdge(p=cur_mp_edge_mask_rate)
        for i in range(len(masked_gs)):
            masked_gs[i] = drop_edge(masked_gs[i])
            masked_gs[i] = dgl.add_self_loop(masked_gs[i])  # we need to add self loop
        enc_emb, _ = self.encoder(masked_gs, feat)
        emb_mapped = self.encoder_to_decoder_edge_recon(enc_emb)

        feat_recon, att_mp = self.decoder(masked_gs, emb_mapped)

        gs_recon = torch.mm(feat_recon, feat_recon.T)
        # ???????????????
        # 为什么都是同一个gs_recon
        loss = att_mp[0] * self.mp_edge_recon_loss(gs_recon, mps[0].to_dense())
        if len(mps) > 1:
            for i in range(len(mps)):
                loss = att_mp[i] * self.mp_edge_recon_loss(gs_recon, mps[i].to_dense())
        return loss

    def encoding_mask_noise(self, feat, node_mask_rate=0.3):
        # We first sample a percentage of nodes from target node type ``self.category``, with node_mask_rate.
        # Specifically, we first replace a percentage of mask tokens
        # by random tokens, with the feat_replace_rate. In addition,
        # we select another percentage of nodes with feat_unchanged_rate and
        # leave them unchanged by utilizing the origin attribute xv,

        # mask: set nodes to 0.0
        # replace: replace nodes with random tokens
        # keep: leave nodes unchanged, remaining origin attr xv

        num_nodes = feat.shape[0]
        all_indices = torch.randperm(num_nodes)
        num_mask_nodes = int(node_mask_rate * num_nodes)
        mask_indices = all_indices[:num_mask_nodes]
        keep_indices = all_indices[num_mask_nodes:]

        # perm_mask = torch.randperm(num_mask_nodes)
        num_unchanged_nodes = int(self.feat_unchanged_rate * num_mask_nodes)
        num_noise_nodes = int(self.feat_replace_rate * num_mask_nodes)

        num_real_mask_nodes = num_mask_nodes - num_unchanged_nodes - num_noise_nodes

        #
        # token_nodes = mask_indices[perm_mask[: num_real_mask_nodes]]
        # noise_nodes = mask_indices[perm_mask[-num_noise_nodes:]]

        token_nodes = mask_indices[: num_real_mask_nodes]
        noise_nodes = mask_indices[-num_noise_nodes:]
        nodes_as_noise = torch.randperm(num_nodes)[:num_noise_nodes]

        out_feat=feat.clone()
        out_feat[token_nodes]=0.0
        out_feat[token_nodes]+=self.enc_mask_token
        if num_nodes > 0:
            out_feat[noise_nodes]=feat[nodes_as_noise]

        return out_feat,(mask_indices,keep_indices)

    def mask_attr_restoration(self, gs, feat, epoch):
        cur_node_mask_rate = self.get_mask_rate(self.node_mask_rate, epoch=epoch)
        use_feat, (mask_nodes, keep_nodes) = self.encoding_mask_noise(feat, cur_node_mask_rate)
        enc_emb, _ = self.encoder(gs, use_feat)
        emb_mapped=self.encoder_to_decoder_attr_restore(enc_emb)

        # we apply another mask token[DM] to H3 before sending it into the decoder. TODO: learnable?
        emb_mapped[mask_nodes]=0.0
        feat_recon, att_mp = self.decoder(gs, emb_mapped)

        feat_before_mask=feat[mask_nodes]
        feat_after_mask=feat_recon[mask_nodes]

        loss = self.attr_restoration_loss(feat_before_mask, feat_after_mask)
        return loss

    def forward(self, hg: dgl.DGLHeteroGraph, h_dict, epoch):

        feat = h_dict[self.category]
        mps = self.get_mps(hg)
        gs = self.mps_to_gs(mps)

        loss = self.mp_edge_recon_loss_weight * self.mask_mp_edge_reconstruction(mps, feat, epoch)

        return loss
