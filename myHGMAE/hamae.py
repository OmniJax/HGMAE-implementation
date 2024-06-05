import torch
import torch.nn as nn
from openhgnn.models import BaseModel

import Models


class HGMAE(BaseModel):
    @classmethod
    def build_model_from_args(cls, args, hg):
        # return cls(
        #
        # )

        pass

    def __init__(self, metapath_dict,
                 in_dim, hidden_dim, num_layers, num_heads, num_out_heads, activation,
                 feat_drop, attn_drop, negative_slope, residual, norm, concat_out,
                 ):
        r'''
        Parameter
        ----------
        metapath_dict: dict[str, list[etype]]
            Dict from meta path name to meta path
        in_dim:
        hidden_dim: dim of encoded embedding
        out_dim:
        num_layers:
        num_heads:
        num_out_heads:
        activation:
        feat_drop:
        attn_drop:
        negative_slope:
        residual:
        norm:
        concat_out:
        encoding:
        '''
        super(HGMAE, self).__init__()
        self.metapth_dict = metapath_dict
        self.num_metapath = len(metapath_dict)
        self.in_dim = in_dim # original feat dim
        self.hidden_dim = hidden_dim # emb dim
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

        assert self.hidden_dim % self.num_heads == 0
        assert self.hidden_dim % self.num_out_heads == 0

        self.enc_dec_input_dim = self.in_dim
        #
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
        self.encoder=Models.HAN(
            num_metapath=self.num_metapath,
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
        self.decoder=Models.HAN(
            num_metapath=self.num_metapath,
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

        # Metapath-based Edge Reconstruction
        self.encoder_to_decoder_edge_recon = nn.Linear(dec_in_dim, dec_in_dim, bias=False)





    def forward(self, *args):
        pass





