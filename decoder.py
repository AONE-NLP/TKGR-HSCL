import os as _os, math as _mh, random as _rd, numpy as _np
import torch, torch.nn as _nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F   
path_dir = _os.getcwd()                

_eps = 1e-6
def _g(t, d, dt, n):
    if not isinstance(t, torch.Tensor):
        raise TypeError(f'{n} must be Tensor')
    if t.dim() < d:
        raise ValueError(f'{n} dim<{d}')
    if t.dtype != dt:
        raise TypeError(f'{n} dtype!={dt}')
    if torch.isinf(t).any() or torch.isnan(t).any():
        t.clamp_(min=-1e4, max=1e4)  

def _s(x):                          
    return F.relu(x.clamp(min=-50, max=50))

class ConvTransR(_nn.Module):
    def __init__(self, num_relations, embedding_dim,
                 input_dropout=0., hidden_dropout=0., feature_map_dropout=0.,
                 channels=50, kernel_size=3, use_bias=True):
        super().__init__()
        self.inp_drop = _nn.Dropout(input_dropout)
        self.hidden_drop = _nn.Dropout(hidden_dropout)
        self.feature_map_drop = _nn.Dropout(feature_map_dropout)
        self.loss = _nn.BCELoss()

        pad = int(_mh.floor(kernel_size / 2))
        self.conv1 = _nn.Conv1d(2, channels, kernel_size, 1, padding=pad, bias=use_bias)
        self.bn0 = _nn.BatchNorm1d(2)
        self.bn1 = _nn.BatchNorm1d(channels)
        self.bn2 = _nn.BatchNorm1d(embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_relations * 2)))
        self.fc = _nn.Linear(embedding_dim * channels, embedding_dim, bias=use_bias)
        self.bn3 = _nn.BatchNorm1d(embedding_dim)   
        self.bn_init = _nn.BatchNorm1d(embedding_dim)

    def forward(self, embedding, emb_rel, triplets,
                nodes_id=None, mode="train", negative_rate=0):
        _g(embedding, 2, torch.float32, 'embedding')
        _g(emb_rel,   2, torch.float32, 'emb_rel')
        _g(triplets,  2, torch.long,    'triplets')

        batch_size = triplets.size(0)
        e1_embedded_all = embedding
        e1_embedded = e1_embedded_all[triplets[:, 0]].unsqueeze(1)
        e2_embedded = e1_embedded_all[triplets[:, 2]].unsqueeze(1)

        x = torch.cat([e1_embedded, e2_embedded], 1)
        x = self.bn0(x)
        x = self.inp_drop(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = _s(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = _s(x)
        return torch.mm(x, emb_rel.t())   

class ConvTransE(_nn.Module):
    def __init__(self, num_entities, embedding_dim,
                 input_dropout=0., hidden_dropout=0., feature_map_dropout=0.,
                 channels=50, kernel_size=3, use_bias=True):
        super().__init__()
        self.inp_drop = _nn.Dropout(input_dropout)
        self.hidden_drop = _nn.Dropout(hidden_dropout)
        self.feature_map_drop = _nn.Dropout(feature_map_dropout)
        self.loss = _nn.CrossEntropyLoss()

        self.gate_linear = _nn.Linear(embedding_dim * 2, embedding_dim, bias=use_bias)
        pad = int(_mh.floor(kernel_size / 2))
        self.conv1 = _nn.Conv1d(2, channels, kernel_size, 1, padding=pad, bias=use_bias)
        self.bn0 = _nn.BatchNorm1d(2)
        self.bn1 = _nn.BatchNorm1d(channels)
        self.bn2 = _nn.BatchNorm1d(embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = _nn.Linear(embedding_dim * channels, embedding_dim, bias=use_bias)
        self.bn3 = _nn.BatchNorm1d(embedding_dim)
        self.bn_init = _nn.BatchNorm1d(embedding_dim)
        self.mlp = _nn.Sequential(
            _nn.Linear(800, 400), _nn.ReLU(),
            _nn.Linear(400, 200)
        )


    def forward(self, embedding, gl_output, lg_output,
                emb_rel, triplets, his_emb,
                pre_weight, pre_type, partial_embeding=None):

        _g(embedding, 2, torch.float32, 'embedding')
        _g(emb_rel,   2, torch.float32, 'emb_rel')
        _g(triplets,  2, torch.long,    'triplets')

        batch_size = triplets.size(0)
        device     = embedding.device
        e1_embedded_all = torch.tanh(embedding)      

        if pre_type == "all":
            idx      = triplets[:, 0]                  
            e1_base  = e1_embedded_all[idx]            
            e1_hist  = torch.tanh(his_emb)[idx]        
            e1_gl    = gl_output[idx]                  
            e1_lg    = lg_output[idx]                  

            gate_bh   = torch.sigmoid(self.gate_linear(
                            torch.cat([e1_base, e1_hist], dim=1)))   
            mix_bh    = gate_bh * e1_base + (1. - gate_bh) * e1_hist

            if pre_weight is None or not (0. <= float(pre_weight) <= 1.):
                pw = 0.5
            else:
                pw = float(pre_weight)
            mix_gl = pw * e1_gl + (1. - pw) * e1_lg

            e1_embed = (mix_bh + mix_gl).unsqueeze(1)  
        else:
            e1_embed = e1_embedded_all[triplets[:, 0]].unsqueeze(1)

        rel_embed = emb_rel[triplets[:, 1]].unsqueeze(1)            
        x = torch.cat([e1_embed, rel_embed], 1)     
        x = self.bn0(x)
        x = self.inp_drop(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = _s(x)
        x = self.feature_map_drop(x)
        x = x.reshape(batch_size, -1)               
        x = self.fc(x)
        x = self.hidden_drop(x)
        if batch_size > 1:
            x = self.bn2(x)
        x = _s(x)
        cl_x = x                                    

        target_mat = embedding if partial_embeding is None else partial_embeding
        out = torch.mm(x, target_mat.t())           

        return out, cl_x



    def forward_slow(self, embedding, emb_rel, triplets):
        e1_all = torch.tanh(embedding)
        batch_size = triplets.size(0)
        e1_emb = e1_all[triplets[:, 0]].unsqueeze(1)
        rel_emb = emb_rel[triplets[:, 1]].unsqueeze(1)
        x = self.bn0(torch.cat([e1_emb, rel_emb], 1))
        x = self.conv1(self.inp_drop(x))
        x = _s(self.bn1(x))
        x = self.feature_map_drop(x).view(batch_size, -1)
        x = _s(self.bn2(self.hidden_drop(self.fc(x)))) if batch_size > 1 else _s(self.fc(x))
        e2_emb = e1_all[triplets[:, 2]]
        return torch.sum(x * e2_emb, dim=1)
