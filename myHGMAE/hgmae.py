# 重写了HAN

import torch
import torch.nn as nn
from openhgnn.models import BaseModel
import dgl
import dgl.sparse.sparse_matrix as sp

import Models
from dgl import DropEdge
from functools import partial
import torch.nn.functional as F
from torch.optim import SparseAdam
from torch.utils.data import DataLoader
from dgl.nn.pytorch import MetaPath2Vec
from tqdm import tqdm
import os

os.environ["DGLBACKEND"] = "pytorch"


def sce_loss(x, y, gamma=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (1 - (x * y).sum(dim=-1)).pow_(gamma)

    loss = loss.mean()
    return loss


class HGMAE(BaseModel):
    r"""

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
        Dropout rate on feature. Default: ``0``
    attn_drop : float, optional
        Dropout rate on attention weight. Default: ``0``
    negative_slope : float, optional
        LeakyReLU angle of negative slope. Defaults: ``0.2``.
    residual : bool, optional
        If True, use residual connection. Defaults: ``False``.

    mp_edge_recon_loss_weight : float
        Trade-off weights for balancing mp_edge_recon_loss. Defaults: ``1.0``
    mp_edge_mask_rate : float
        Metapath-based edge masking rate. Defaults: ``0.6``
    mp_edge_gamma : float
        Scaling factor of mp_edge_recon_loss when using ``sce`` as loss function. Defaults: ``3.0``

    node_mask_rate : str
        Linearly increasing attribute mask rate to sample a subset of nodes, in the format of 'min,delta,max'. Defaults: ``'0.5,0.005,0.8'``
    attr_restore_loss_weight : float
        Trade-off weights for balancing attr_restore_loss. Defaults: ``1.0``
    attr_restore_gamma : float
        Scaling factor of att_restore_loss when using ``sce`` as loss function. Defaults: ``1.0``
    attr_replace_rate : float
        Replacing a percentage of mask tokens by random tokens, with the attr_replace_rate. Defaults: ``0.3``
    attr_unchanged_rate : float
        Leaving a percentage of nodes unchanged by utilizing the origin attribute, with the attr_unchanged_rate. Defaults: ``0.2``
    mp2vec_window_size : int
        In a random walk :attr:`w`, a node :attr:`w[j]` is considered close to a node :attr:`w[i]` if :attr:`i - window_size <= j <= i + window_size`. Defaults: ``3``
    mp2vec_rw_length : int
        The length of each random walk. Defaults: ``10``
    mp2vec_walks_per_node=args.mp2vec_walks_per_node,
        The number of walks to sample for each node. Defaults: ``2``

    mp2vec_negative_size: int
        Number of negative samples to use for each positive sample. Default: ``5``
    mp2vec_batch_size : int
        How many samples per batch to load when training mp2vec_feat. Defaults: ``128``
    mp2vec_train_epoch : int
        The training epochs of MetaPath2Vec model. Default: ``20``
    mp2vec_train_lr : float
        The training learning rate of MetaPath2Vec model. Default: ``0.001``
    mp2vec_feat_dim : int
        The feature dimension of MetaPath2Vec model. Defaults: ``128``
    mp2vec_feat_pred_loss_weight : float
        Trade-off weights for balancing mp2vec_feat_pred_loss. Defaults: ``0.1``
    mp2vec_feat_gamma: flaot
        Scaling factor of mp2vec_feat_pred_loss when using ``sce`` as loss function. Defaults: ``2.0``
    mp2vec_feat_drop: float
        The dropout rate of self.enc_out_to_mp2vec_feat_mapping. Defaults: ``0.2``
    """

    @classmethod
    def build_model_from_args(cls, args, hg, metapaths_dict: dict):
        return cls(
            hg=hg,
            metapaths_dict=metapaths_dict,
            category=args.category,
            in_dim=args.in_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            num_out_heads=args.num_out_heads,
            feat_drop=args.feat_drop,
            attn_drop=args.attn_drop,
            negative_slope=args.negative_slope,
            residual=args.residual,

            # Metapath-based Edge Reconstruction
            mp_edge_recon_loss_weight=args.mp_edge_recon_loss_weight,
            mp_edge_mask_rate=args.mp_edge_mask_rate,
            mp_edge_gamma=args.mp_edge_gamma,

            # Type-specific Attribute Restoration
            node_mask_rate=args.node_mask_rate,
            attr_restore_gamma=args.attr_restore_gamma,
            attr_restore_loss_weight=args.attr_restore_loss_weight,
            attr_replace_rate=args.attr_replace_rate,
            attr_unchanged_rate=args.attr_unchanged_rate,

            # Positional Feature Prediction
            mp2vec_negative_size=args.mp2vec_negative_size,
            mp2vec_window_size=args.mp2vec_window_size,
            mp2vec_rw_length=args.mp2vec_rw_length,
            mp2vec_walks_per_node=args.mp2vec_walks_per_node,
            mp2vec_batch_size=args.mp2vec_batch_size,
            mp2vec_train_epoch=args.mp2vec_train_epoch,
            mp2vec_trian_lr=args.mp2vec_train_lr,
            mp2vec_feat_dim=args.mp2vec_feat_dim,
            mp2vec_feat_pred_loss_weight=args.mp2vec_feat_pred_loss_weight,
            mp2vec_feat_gamma=args.mp2vec_feat_gamma,
            mp2vec_feat_drop=args.mp2vec_feat_drop,

        )

    def __init__(self, hg, metapaths_dict, category,
                 in_dim, hidden_dim, num_layers, num_heads, num_out_heads,
                 feat_drop=0, attn_drop=0, negative_slope=0.2, residual=False,
                 mp_edge_recon_loss_weight=1, mp_edge_mask_rate=0.6, mp_edge_gamma=3,
                 attr_restore_loss_weight=1, attr_restore_gamma=1, node_mask_rate='0.5,0.005,0.8',
                 attr_replace_rate=0.3, attr_unchanged_rate=0.2,
                 mp2vec_window_size=3, mp2vec_negative_size=5, mp2vec_rw_length=10, mp2vec_walks_per_node=2,
                 mp2vec_feat_dim=128, mp2vec_feat_drop=0.2,
                 mp2vec_train_epoch=20, mp2vec_batch_size=128, mp2vec_trian_lr=0.001,
                 mp2vec_feat_pred_loss_weight=0.1, mp2vec_feat_gamma=2
                 ):
        super(HGMAE, self).__init__()
        # self.device = hg.device
        # self.device = None
        self.metapaths_dict = metapaths_dict
        self.num_metapaths = len(metapaths_dict)
        self.category = category
        self.in_dim = in_dim  # original feat dim
        self.hidden_dim = hidden_dim  # emb dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_out_heads = num_out_heads
        self.feat_drop = feat_drop
        self.attn_drop = attn_drop
        self.negative_slope = negative_slope
        self.residual = residual
        # self.norm = norm
        # self.concat_out = concat_out
        # self.loss_func = loss_func

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
            feat_drop=self.feat_drop,
            attn_drop=self.attn_drop,
            negative_slope=self.negative_slope,
            residual=self.residual,
            # norm=self.norm,
            # concat_out=self.concat_out,
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
            feat_drop=self.feat_drop,
            attn_drop=self.attn_drop,
            negative_slope=self.negative_slope,
            residual=self.residual,
            # norm=self.norm,
            # concat_out=self.concat_out,
            encoding=False
        )
        self.__cached_gs = None  # cached metapath reachable graphs
        self.__cached_mps = None  # cached metapath adjacency matrices (SparseMatrix)

        # Metapath-based Edge Reconstruction
        self.mp_edge_recon_loss_weight = mp_edge_recon_loss_weight
        self.mp_edge_mask_rate = mp_edge_mask_rate
        self.mp_edge_gamma = mp_edge_gamma
        self.mp_edge_recon_loss = partial(sce_loss, gamma=mp_edge_gamma)
        self.encoder_to_decoder_edge_recon = nn.Linear(dec_in_dim, dec_in_dim, bias=False)

        # Type-specific Attribute Restoration
        self.attr_restore_gamma = attr_restore_gamma
        self.enc_mask_token = nn.Parameter(torch.zeros(1, self.enc_dec_input_dim))  # learnable mask token [M]
        self.encoder_to_decoder_attr_restore = nn.Linear(dec_in_dim, dec_in_dim, bias=False)
        self.attr_restore_loss = partial(sce_loss, gamma=attr_restore_gamma)
        self.attr_restore_loss_weight = attr_restore_loss_weight
        self.node_mask_rate = node_mask_rate

        assert attr_replace_rate + attr_unchanged_rate < 1, "attr_replace_rate + attr_unchanged_rate must " \
                                                            "be smaller than 1 "
        self.attr_unchanged_rate = attr_unchanged_rate
        self.attr_replace_rate = attr_replace_rate

        # Positional Feature Prediction
        self.mp2vec_feat_dim = mp2vec_feat_dim
        self.mp2vec_window_size = mp2vec_window_size
        self.mp2vec_negative_size = mp2vec_negative_size
        self.mp2vec_batch_size = mp2vec_batch_size
        self.mp2vec_train_lr = mp2vec_trian_lr
        self.mp2vec_train_epoch = mp2vec_train_epoch
        self.mp2vec_walks_per_node = mp2vec_walks_per_node
        self.mp2vec_rw_length = mp2vec_rw_length

        self.mp2vec_feat = None
        self.mp2vec_feat_pred_loss_weight = mp2vec_feat_pred_loss_weight
        self.mp2vec_feat_drop = mp2vec_feat_drop
        self.mp2vec_feat_gamma = mp2vec_feat_gamma
        self.mp2vec_feat_pred_loss = partial(sce_loss, gamma=self.mp2vec_feat_gamma)

        self.enc_out_to_mp2vec_feat_mapping = nn.Sequential(
            nn.Linear(dec_in_dim, self.mp2vec_feat_dim),
            nn.PReLU(),
            nn.Dropout(self.mp2vec_feat_drop),
            nn.Linear(self.mp2vec_feat_dim, self.mp2vec_feat_dim),
            nn.PReLU(),
            nn.Dropout(self.mp2vec_feat_drop),
            nn.Linear(self.mp2vec_feat_dim, self.mp2vec_feat_dim)
        )

    def train_mp2vec(self, hg):
        device = hg.device
        num_nodes = hg.num_nodes(self.category)

        # metapath for metapath2vec model
        Mp4Mp2Vec = []
        mp_nodes_seq = []
        for mp_name, mp in self.metapaths_dict.items():
            Mp4Mp2Vec += mp
            assert (mp[0][0] == mp[-1][-1]), "The start node type and the end one in metapath should be the same."

        x = max(self.mp2vec_rw_length // (len(Mp4Mp2Vec) + 1), 1)
        Mp4Mp2Vec *= x
        for mp in Mp4Mp2Vec:
            mp_nodes_seq.append(mp[0])
        mp_nodes_seq.append(mp[-1])
        assert (
                mp_nodes_seq[0] == mp_nodes_seq[-1]
        ), "The start node type and the end one in metapath should be the same."
        print("Metapath for training mp2vec models:", mp_nodes_seq)
        m2v_model = MetaPath2Vec(
            hg, Mp4Mp2Vec, self.mp2vec_window_size, self.mp2vec_feat_dim, self.mp2vec_negative_size
        ).to(device)
        m2v_model.train()
        dataloader = DataLoader(
            list(range(num_nodes)) * self.mp2vec_walks_per_node,
            batch_size=self.mp2vec_batch_size,
            shuffle=True,
            collate_fn=m2v_model.sample,
        )
        optimizer = SparseAdam(m2v_model.parameters(), lr=self.mp2vec_train_lr)
        for _ in tqdm(range(self.mp2vec_train_epoch)):
            for pos_u, pos_v, neg_v in dataloader:
                loss = m2v_model(pos_u.to(device), pos_v.to(device), neg_v.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # get the embeddings
        nids = torch.LongTensor(m2v_model.local_to_global_nid[self.category]).to(device)
        emb = m2v_model.node_embed(nids)

        del m2v_model, nids, pos_u, pos_v, neg_v
        if device == "cuda":
            torch.cuda.empty_cache()
        return emb.detach()

    def get_mask_rate(self, input_mask_rate, get_min=False, epoch=None):
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

    def normalize_feat(self, feat):
        rowsum = torch.sum(feat, dim=1).reshape(-1, 1)
        r_inv = torch.pow(rowsum, -1)
        r_inv = torch.where(torch.isinf(r_inv), 0, r_inv)
        return feat * r_inv

    def normalize_adj(self, adj):
        rowsum = torch.sum(adj, dim=1).reshape(-1, 1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt = torch.where(torch.isinf(d_inv_sqrt), 0, d_inv_sqrt)
        return d_inv_sqrt * adj.T * d_inv_sqrt.T  # T?

    def get_mps(self, hg: dgl.DGLHeteroGraph):
        # mps: a list of metapath-based adjacency matrices (SparseMatrix)
        if self.__cached_mps is None:
            self.__cached_mps = []
            for mp in self.metapaths_dict.values():
                adj = dgl.metapath_reachable_graph(hg, mp).adjacency_matrix()
                adj = self.normalize_adj(adj.to_dense()).to_sparse()  # torch_sparse
                # adj = sp.from_torch_sparse(adj)
                self.__cached_mps.append(adj)
            # self.__cached_mps = [dgl.metapath_reachable_graph(hg, mp).adjacency_matrix() for mp in
            #                      self.metapaths_dict.values()]
        return self.__cached_mps.copy()

    def mps_to_gs(self, mps):
        # gs: a list of meta path reachable graphs that only contain topological structures
        # without edge and node features
        if self.__cached_gs is None:
            gs = []
            for mp in mps:
                indices = mp.indices()
                cur_graph = dgl.graph((indices[0], indices[1]))
                cur_graph = dgl.add_self_loop(cur_graph)  # we need to add self loop
                gs.append(cur_graph)
            self.__cached_gs = gs
        return self.__cached_gs.copy()

    def mask_mp_edge_reconstruction(self, mps, feat, epoch):
        masked_gs = self.mps_to_gs(mps)
        cur_mp_edge_mask_rate = self.get_mask_rate(self.mp_edge_mask_rate, epoch=epoch)
        drop_edge = DropEdge(p=cur_mp_edge_mask_rate)
        for i in range(len(masked_gs)):
            masked_gs[i] = drop_edge(masked_gs[i])
            masked_gs[i] = dgl.add_self_loop(masked_gs[i])  # we need to add self loop
        enc_emb, _ = self.encoder(masked_gs, feat)
        emb_mapped = self.encoder_to_decoder_edge_recon(enc_emb)

        feat_recon, att_mp = self.decoder(masked_gs, emb_mapped)
        gs_recon = torch.mm(feat_recon,feat_recon.T )

        # ??????为什么都是同一个gs_recon
        loss = att_mp[0] * self.mp_edge_recon_loss(gs_recon, mps[0].to_dense())
        if len(mps) > 1:
            for i in range(len(mps)):
                loss = att_mp[i] * self.mp_edge_recon_loss(gs_recon, mps[i].to_dense())
        return loss

    def encoding_mask_noise(self, feat, node_mask_rate=0.3):
        # We first sample a percentage of nodes from target node type ``self.category``, with node_mask_rate.
        # Specifically, we first replace a percentage of mask tokens
        # by random tokens, with the attr_replace_rate. In addition,
        # we select another percentage of nodes with attr_unchanged_rate and
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
        num_unchanged_nodes = int(self.attr_unchanged_rate * num_mask_nodes)
        num_noise_nodes = int(self.attr_replace_rate * num_mask_nodes)

        num_real_mask_nodes = num_mask_nodes - num_unchanged_nodes - num_noise_nodes

        #
        # token_nodes = mask_indices[perm_mask[: num_real_mask_nodes]]
        # noise_nodes = mask_indices[perm_mask[-num_noise_nodes:]]

        token_nodes = mask_indices[: num_real_mask_nodes]
        noise_nodes = mask_indices[-num_noise_nodes:]
        nodes_as_noise = torch.randperm(num_nodes)[:num_noise_nodes]

        out_feat = feat.clone()
        out_feat[token_nodes] = 0.0
        out_feat[token_nodes] += self.enc_mask_token
        if num_nodes > 0:
            out_feat[noise_nodes] = feat[nodes_as_noise]
            
        return out_feat, (mask_indices, keep_indices)

    def mask_attr_restoration(self, gs, feat, epoch):
        cur_node_mask_rate = self.get_mask_rate(self.node_mask_rate, epoch=epoch)
        use_feat, (mask_nodes, keep_nodes) = self.encoding_mask_noise(feat, cur_node_mask_rate)
        enc_emb, _ = self.encoder(gs, use_feat)  # H3
        emb_mapped = self.encoder_to_decoder_attr_restore(enc_emb)

        # we apply another mask token[DM] to H3 before sending it into the decoder. TODO: learnable?
        emb_mapped[mask_nodes] = 0.0
        feat_recon, att_mp = self.decoder(gs, emb_mapped)

        feat_before_mask = feat[mask_nodes]
        feat_after_mask = feat_recon[mask_nodes]

        loss = self.attr_restore_loss(feat_before_mask, feat_after_mask)
        return loss, enc_emb

    def forward(self, hg: dgl.heterograph, h_dict, trained_mp2vec_feat_dict=None, epoch=None):
        assert epoch is not None, "epoch should be a positive integer"
        # TODO: 源码加了preprocess_features，
        if trained_mp2vec_feat_dict is None:
            if self.mp2vec_feat is None:
                print("Training MetaPath2Vec feat by given metapaths_dict ")
                self.mp2vec_feat = self.train_mp2vec(hg)
                self.mp2vec_feat = self.normalize_feat(self.mp2vec_feat)
            mp2vec_feat = self.mp2vec_feat
        else:
            mp2vec_feat = trained_mp2vec_feat_dict[self.category]
        mp2vec_feat = mp2vec_feat.to(hg.device)

        feat = h_dict[self.category].to(hg.device)
        feat = self.normalize_feat(feat)
        mps = self.get_mps(hg)
        gs = self.mps_to_gs(mps)

        # MER
        mp_edge_recon_loss = self.mp_edge_recon_loss_weight * self.mask_mp_edge_reconstruction(mps, feat, epoch)
        # print(mp_edge_recon_loss.detach())

        # TAR
        attr_restore_loss, enc_emb = self.mask_attr_restoration(gs, feat, epoch)
        attr_restore_loss *= self.attr_restore_loss_weight
        # print(attr_restore_loss.detach())

        # PFP
        mp2vec_feat_pred = self.enc_out_to_mp2vec_feat_mapping(enc_emb)  # H3
        mp2vec_feat_pred_loss = self.mp2vec_feat_pred_loss_weight * self.mp2vec_feat_pred_loss(mp2vec_feat_pred,
                                                                                               mp2vec_feat)

        loss = mp_edge_recon_loss + attr_restore_loss + mp2vec_feat_pred_loss

        return loss

    def get_mp2vec_feat(self):
        return self.mp2vec_feat

    def get_embeds(self, hg, h_dict):
        with torch.no_grad():
            self.eval()
            feat = h_dict[self.category].to(hg.device)
            mps = self.get_mps(hg)
            gs = self.mps_to_gs(mps)
            emb, _ = self.encoder(gs, feat)
            return emb.detach()


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret
