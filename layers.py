import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch.softmax import edge_softmax
import numpy as np


def _maybe_dropout(x, drop):
    return drop(x) if drop is not None else x


class RGCNLayer(nn.Module):
    def __init__(
        self,
        in_feat,
        out_feat,
        bias=None,
        activation=None,
        self_loop=False,
        skip_connect=False,
        dropout=0.0,
        layer_norm=False,
    ):
        super().__init__()
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.skip_connect = skip_connect
        self.layer_norm = layer_norm
        if bias:
            self.bias = nn.Parameter(torch.empty(out_feat))
            nn.init.xavier_uniform_(self.bias, gain=nn.init.calculate_gain("relu"))
        if self_loop:
            self.loop_weight = nn.Parameter(torch.empty(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain("relu"))
        if skip_connect:
            self.skip_connect_weight = nn.Parameter(torch.empty(out_feat, out_feat))
            nn.init.xavier_uniform_(self.skip_connect_weight, gain=nn.init.calculate_gain("relu"))
            self.skip_connect_bias = nn.Parameter(torch.zeros(out_feat))
        self.dropout = nn.Dropout(dropout) if dropout else None
        if layer_norm:
            self.normalization_layer = nn.LayerNorm(out_feat, elementwise_affine=False)

    def propagate(self, g):
        raise NotImplementedError

    def forward(self, g, prev_h=None):
        if prev_h is None:
            prev_h = []
        if self.self_loop:
            loop_message = _maybe_dropout(torch.mm(g.ndata["h"], self.loop_weight), self.dropout)
        if prev_h and self.skip_connect:
            skip_weight = torch.sigmoid(torch.mm(prev_h, self.skip_connect_weight) + self.skip_connect_bias)
        self.propagate(g)
        h = g.ndata["h"]
        if self.bias is not None:
            h = h + self.bias
        if prev_h and self.skip_connect:
            if self.activation is not None:
                h = self.activation(h)
            if self.self_loop:
                loop_part = self.activation(loop_message) if self.activation else loop_message
                h = h + skip_weight * loop_part
            h = h + (1 - skip_weight) * prev_h
        else:
            if self.self_loop:
                h = h + loop_message
            if self.layer_norm:
                h = self.normalization_layer(h)
            if self.activation is not None:
                h = self.activation(h)
        g.ndata["h"] = h
        return h


class RGCNBasisLayer(RGCNLayer):
    def __init__(
        self,
        in_feat,
        out_feat,
        num_rels,
        num_bases=-1,
        bias=None,
        activation=None,
        is_input_layer=False,
    ):
        super().__init__(in_feat, out_feat, bias, activation)
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.num_bases = num_bases if 0 < num_bases <= num_rels else num_rels
        self.is_input_layer = is_input_layer
        self.weight = nn.Parameter(torch.empty(self.num_bases, in_feat, out_feat))
        if self.num_bases < num_rels:
            self.w_comp = nn.Parameter(torch.empty(num_rels, self.num_bases))
            nn.init.xavier_uniform_(self.w_comp, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain("relu"))

    def propagate(self, g):
        if self.num_bases < self.num_rels:
            w = self.weight.view(self.num_bases, self.in_feat * self.out_feat)
            w = torch.matmul(self.w_comp, w).view(self.num_rels, self.in_feat, self.out_feat)
        else:
            w = self.weight

        if self.is_input_layer:

            def msg(edges):
                idx = edges.data["type"] * self.in_feat + edges.src["id"]
                return {"msg": w.view(-1, self.out_feat).index_select(0, idx)}

        else:

            def msg(edges):
                ew = w.index_select(0, edges.data["type"])
                m = torch.bmm(edges.src["h"].unsqueeze(1), ew).squeeze()
                return {"msg": m}

        def apply(nodes):
            return {"h": nodes.data["h"] * nodes.data["norm"]}

        g.update_all(msg, fn.sum(msg="msg", out="h"), apply)


class RGCNBlockLayer(RGCNLayer):
    def __init__(
        self,
        in_feat,
        out_feat,
        num_rels,
        num_bases,
        bias=None,
        activation=None,
        self_loop=False,
        dropout=0.0,
        skip_connect=False,
        layer_norm=False,
    ):
        super().__init__(
            in_feat,
            out_feat,
            bias,
            activation,
            self_loop=self_loop,
            skip_connect=skip_connect,
            dropout=dropout,
            layer_norm=layer_norm,
        )
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.sub_in = in_feat // num_bases
        self.sub_out = out_feat // num_bases
        self.weight = nn.Parameter(torch.empty(num_rels, num_bases * self.sub_in * self.sub_out))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain("relu"))

    def msg_func(self, edges):
        w = self.weight.index_select(0, edges.data["type"]).view(-1, self.sub_in, self.sub_out)
        node = edges.src["h"].view(-1, 1, self.sub_in)
        m = torch.bmm(node, w).view(-1, self.out_feat)
        return {"msg": m}

    @staticmethod
    def apply_func(nodes):
        return {"h": nodes.data["h"] * nodes.data["norm"]}

    def propagate(self, g):
        g.update_all(self.msg_func, fn.sum(msg="msg", out="h"), self.apply_func)


class UnionRGCNLayer(RGCNLayer):
    def __init__(
        self,
        in_feat,
        out_feat,
        num_rels,
        num_bases=-1,
        bias=None,
        activation=None,
        self_loop=False,
        dropout=0.0,
        skip_connect=False,
        rel_emb=None,
    ):
        super().__init__(
            in_feat,
            out_feat,
            bias,
            activation,
            self_loop=self_loop,
            skip_connect=skip_connect,
            dropout=dropout,
        )
        self.num_rels = num_rels
        self.rel_emb = None
        self.weight_neighbor = nn.Parameter(torch.empty(in_feat, out_feat))
        nn.init.xavier_uniform_(self.weight_neighbor, gain=nn.init.calculate_gain("relu"))
        if self_loop:
            self.evolve_loop_weight = nn.Parameter(torch.empty(in_feat, out_feat))
            nn.init.xavier_uniform_(self.evolve_loop_weight, gain=nn.init.calculate_gain("relu"))

    def propagate(self, g):
        g.update_all(self.msg_func, fn.sum(msg="msg", out="h"), self.apply_func)

    def forward(self, g, prev_h, emb_rel):
        self.rel_emb = emb_rel
        if self.self_loop:
            mask = (g.in_degrees() > 0).to(g.ndata["h"].device)
            loop = torch.mm(g.ndata["h"], self.evolve_loop_weight)
            loop[mask] = torch.mm(g.ndata["h"], self.loop_weight)[mask]
        if prev_h is not None and self.skip_connect:
            skip_weight = torch.sigmoid(torch.mm(prev_h, self.skip_connect_weight) + self.skip_connect_bias)
        self.propagate(g)
        h = g.ndata["h"]
        if prev_h is not None and self.skip_connect:
            if self.self_loop:
                h = h + loop
            h = skip_weight * h + (1 - skip_weight) * prev_h
        else:
            if self.self_loop:
                h = h + loop
        if self.activation is not None:
            h = self.activation(h)
        h = _maybe_dropout(h, self.dropout)
        g.ndata["h"] = h
        return h

    def msg_func(self, edges):
        r = self.rel_emb.index_select(0, edges.data["type"]).view(-1, self.out_feat)
        m = torch.mm(edges.src["h"].view(-1, self.out_feat) + r, self.weight_neighbor)
        return {"msg": m}

    @staticmethod
    def apply_func(nodes):
        return {"h": nodes.data["h"] * nodes.data["norm"]}


class UnionRGCNLayer2(UnionRGCNLayer):
    pass


class RGAT(nn.Module):
    def __init__(
        self,
        in_feat,
        out_feat,
        bias=None,
        activation=None,
        self_loop=False,
        dropout=0.0,
        layer_norm=False,
    ):
        super().__init__()
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.layer_norm = layer_norm
        if bias:
            self.bias = nn.Parameter(torch.empty(out_feat))
            nn.init.xavier_uniform_(self.bias, gain=nn.init.calculate_gain("relu"))
        if self_loop:
            self.loop_weight = nn.Parameter(torch.empty(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain("relu"))
            self.evolve_loop_weight = nn.Parameter(torch.empty(in_feat, out_feat))
            nn.init.xavier_uniform_(self.evolve_loop_weight, gain=nn.init.calculate_gain("relu"))
        self.dropout = nn.Dropout(dropout) if dropout else None
        if layer_norm:
            self.normalization_layer = nn.LayerNorm(out_feat, elementwise_affine=False)
        self.num_head = 5
        self.out_feat = out_feat
        self.in_feat = in_feat
        self.head_dim = in_feat // self.num_head
        self.W_t = nn.Parameter(torch.empty(out_feat, out_feat))
        nn.init.xavier_uniform_(self.W_t, gain=nn.init.calculate_gain("relu"))
        self.W_r = nn.Parameter(torch.empty(out_feat, out_feat))
        nn.init.xavier_uniform_(self.W_r, gain=nn.init.calculate_gain("relu"))
        self.w_triplet = nn.Parameter(torch.empty(in_feat * 3, out_feat))
        nn.init.xavier_uniform_(self.w_triplet, gain=nn.init.calculate_gain("relu"))
        self.w_quad = nn.Parameter(torch.empty(in_feat, out_feat))
        nn.init.xavier_uniform_(self.w_quad, gain=nn.init.calculate_gain("relu"))

    @staticmethod
    def apply_func(nodes):
        return {"h": nodes.data["h"] * nodes.data["norm"]}

    def quads_msg_func(self, edges):
        triplet = torch.cat([edges.src["h"], edges.data["rel_emb"], edges.dst["h"]], dim=1)
        triplet = torch.mm(triplet, self.w_triplet)
        a_triplet = torch.mm(triplet + edges.data["fre"].unsqueeze(1), self.w_quad)
        return {"triplet": triplet, "a_triplet": a_triplet}

    def msg_func(self, edges):
        return {"msg": edges.data["att_triplet"] * edges.data["triplet"]}

    def propagate(self, g):
        g.apply_edges(self.quads_msg_func)
        g.edata["a_triplet"] = F.leaky_relu(g.edata["a_triplet"])
        g.edata["att_triplet"] = edge_softmax(g, g.edata["a_triplet"])
        g.update_all(self.msg_func, fn.sum(msg="msg", out="h"), self.apply_func)

    def forward(self, g, node, rel):
        g.ndata["h"] = node
        g.edata["rel_emb"] = rel[g.edata["type"]]
        if self.self_loop:
            mask = (g.in_degrees() > 0).to(node.device)
            loop = torch.mm(node, self.evolve_loop_weight)
            loop[mask] = torch.mm(node, self.loop_weight)[mask]
            loop = _maybe_dropout(loop, self.dropout)
        self.propagate(g)
        h = g.ndata["h"]
        if self.self_loop:
            h = h + loop
        g.ndata["h"] = h
        return self.activation(h) if self.activation else h

    def forward_v(self, g, batchsize, node, rel, tim):
        g.ndata["h"] = node
        g.edata["rel_emb"] = rel[g.edata["type"]]
        if self.self_loop:
            loop = _maybe_dropout(node, self.dropout)
        self.propagate(g)
        return self.activation(g.ndata["h"]) if self.activation else g.ndata["h"]

    def forward_v2(self, g, batchsize, node, rel, device):
        g.ndata["h"] = node
        if self.self_loop:
            loop = _maybe_dropout(node, self.dropout)
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        train_nids = np.random.choice(np.arange(g.num_nodes()), g.num_nodes(), replace=False)
        dataloader = dgl.dataloading.DataLoader(g, train_nids, sampler, batch_size=batchsize, device=device)
        b = torch.zeros(g.num_nodes(), self.in_feat, device=device)
        for _, out_nodes, blocks in dataloader:
            blocks[0].edata["rel_emb"] = rel[blocks[0].edata["type"]]
            self.propagate(blocks[0])
        for _, out_nodes, blocks in dataloader:
            b[out_nodes] = blocks[0].dstdata["h"]
        h = b
        if self.self_loop:
            h = h + loop
        g.ndata["h"] = h
        return self.activation(h) if self.activation else h


class UnionRGATLayer(nn.Module):
    def __init__(
        self,
        in_feat,
        out_feat,
        num_rels,
        num_bases=-1,
        bias=None,
        activation=None,
        self_loop=False,
        dropout=0.0,
        skip_connect=False,
        rel_emb=None,
    ):
        super().__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.num_rels = num_rels
        self.rel_emb = None
        self.skip_connect = skip_connect
        self.weight_neighbor = nn.Parameter(torch.empty(in_feat, out_feat))
        nn.init.xavier_uniform_(self.weight_neighbor, gain=nn.init.calculate_gain("relu"))
        if self_loop:
            self.loop_weight = nn.Parameter(torch.empty(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain("relu"))
        if skip_connect:
            self.skip_connect_weight = nn.Parameter(torch.empty(out_feat, out_feat))
            nn.init.xavier_uniform_(self.skip_connect_weight, gain=nn.init.calculate_gain("relu"))
            self.skip_connect_bias = nn.Parameter(torch.zeros(out_feat))
        self.dropout = nn.Dropout(dropout) if dropout else None
        self.attn_fc = nn.Linear(3 * out_feat, out_feat, bias=False)
        self.attn_fc2 = nn.Linear(out_feat, 1, bias=False)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=nn.init.calculate_gain("relu"))

    def edge_attention(self, edges):
        r = self.rel_emb.index_select(0, edges.data["type"]).view(-1, self.out_feat)
        z = torch.cat([edges.src["h"], edges.dst["h"], r], dim=1)
        a = self.attn_fc2(self.attn_fc(z))
        return {"e_att": F.leaky_relu(a)}

    def msg_func(self, edges):
        return {"e_h": edges.src["h"], "e_att": edges.data["e_att"]}

    def reduce_func(self, nodes):
        a = F.softmax(nodes.mailbox["e_att"], dim=1)
        h = torch.sum(a * nodes.mailbox["e_h"], dim=1)
        return {"h": h}

    def propagate(self, g):
        g.update_all(self.msg_func, self.reduce_func)

    def forward(self, g, prev_h, emb_rel):
        self.rel_emb = emb_rel
        if self.self_loop:
            loop = torch.mm(g.ndata["h"], self.loop_weight)
        if prev_h is not None and self.skip_connect:
            skip_weight = torch.sigmoid(torch.mm(prev_h, self.skip_connect_weight) + self.skip_connect_bias)
        g.apply_edges(self.edge_attention)
        self.propagate(g)
        h = g.ndata["h"]
        if prev_h is not None and self.skip_connect:
            if self.self_loop:
                h = h + loop
            h = skip_weight * h + (1 - skip_weight) * prev_h
        else:
            if self.self_loop:
                h = h + loop
        if self.activation is not None:
            h = self.activation(h)
        h = _maybe_dropout(h, self.dropout)
        g.ndata["h"] = h
        return h


class CompGCNLayer(nn.Module):
    def __init__(
        self,
        in_feat,
        out_feat,
        num_rels,
        comp,
        num_bases=-1,
        bias=None,
        activation=None,
        self_loop=False,
        dropout=0.0,
        skip_connect=False,
        rel_emb=None,
    ):
        super().__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.num_rels = num_rels
        self.rel_emb = None
        self.skip_connect = skip_connect
        self.comp = comp
        self.weight_neighbor = nn.Parameter(torch.empty(in_feat, out_feat))
        nn.init.xavier_uniform_(self.weight_neighbor, gain=nn.init.calculate_gain("relu"))
        if self_loop:
            self.loop_weight = nn.Parameter(torch.empty(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain("relu"))
        if skip_connect:
            self.skip_connect_weight = nn.Parameter(torch.empty(out_feat, out_feat))
            nn.init.xavier_uniform_(self.skip_connect_weight, gain=nn.init.calculate_gain("relu"))
            self.skip_connect_bias = nn.Parameter(torch.zeros(out_feat))
        self.dropout = nn.Dropout(dropout) if dropout else None

    @staticmethod
    def apply_func(nodes):
        return {"h": nodes.data["h"] * nodes.data["norm"]}

    def msg_func(self, edges):
        r = self.rel_emb.index_select(0, edges.data["type"]).view(-1, self.out_feat)
        n = edges.src["h"].view(-1, self.out_feat)
        if self.comp == "sub":
            m = n + r
        else:
            m = n * r
        m = torch.mm(m, self.weight_neighbor)
        return {"msg": m}

    def propagate(self, g):
        g.update_all(self.msg_func, fn.sum(msg="msg", out="h"), self.apply_func)

    def forward(self, g, prev_h, emb_rel):
        self.rel_emb = emb_rel
        if self.self_loop:
            loop = torch.mm(g.ndata["h"], self.loop_weight)
        if prev_h is not None and self.skip_connect:
            skip_weight = torch.sigmoid(torch.mm(prev_h, self.skip_connect_weight) + self.skip_connect_bias)
        self.propagate(g)
        h = g.ndata["h"]
        if prev_h is not None and self.skip_connect:
            if self.self_loop:
                h = h + loop
            h = skip_weight * h + (1 - skip_weight) * prev_h
        else:
            if self.self_loop:
                h = h + loop
        if self.activation is not None:
            h = self.activation(h)
        h = _maybe_dropout(h, self.dropout)
        g.ndata["h"] = h
        return h


class UnionRGATLayer2(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1,  bias=None,
                 activation=None, self_loop=False, dropout=0.0, skip_connect=False, rel_emb=None):
        super(UnionRGATLayer2, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.num_rels = num_rels
        self.rel_emb = None
        self.skip_connect = skip_connect
        self.ob = None
        self.sub = None

        # WL
        self.weight_neighbor = nn.Parameter(torch.Tensor(self.in_feat, self.out_feat))
        nn.init.xavier_uniform_(self.weight_neighbor, gain=nn.init.calculate_gain('relu'))

        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
            self.evolve_loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.evolve_loop_weight, gain=nn.init.calculate_gain('relu'))

        if self.skip_connect:
            self.skip_connect_weight = nn.Parameter(torch.Tensor(out_feat, out_feat)) 
            nn.init.xavier_uniform_(self.skip_connect_weight,gain=nn.init.calculate_gain('relu'))
            self.skip_connect_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.skip_connect_bias) 

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        # equation (2)
        self.attn_fc = nn.Linear(3 * self.out_feat, self.out_feat, bias=False)  
        self.attn_fc2 = nn.Linear(self.out_feat, 1, bias=False)  
        # self.leakyrelu = nn.LeakyReLU(0.1)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=nn.init.calculate_gain('relu'))  

    def edge_attention(self, edges):
            # edge UDF for equation (2)
        relation = self.rel_emb.index_select(0, edges.data['type']).view(-1, self.out_feat)
        node_h = edges.src['h'].view(-1, self.out_feat)
        node_t = edges.dst['h'].view(-1, self.out_feat)  
         
        z2 = torch.cat([node_h, node_t, relation], dim=1)
        a = self.attn_fc(z2)
        a = self.attn_fc2(a)
        return {'e_att': F.leaky_relu(a)}

    def propagate(self, g):
        g.update_all(self.msg_func, self.reduce_func)

    def msg_func(self, edges):

        relation = self.rel_emb.index_select(0, edges.data['type']).view(-1, self.out_feat)
        edge_type = edges.data['type']
        edge_num = edge_type.shape[0]
        node = edges.src['h'].view(-1, self.out_feat)
        node_t = edges.dst['h'].view(-1, self.out_feat)

        msg = torch.cat([node, node_t, relation], dim=1)
        msg = self.attn_fc(msg)

        return {'e_h': edges.src['h'], 'e_att': edges.data['e_att']}

    def reduce_func(self, nodes):

        alpha = F.softmax(nodes.mailbox['e_att'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['e_h'], dim=1)
        return {'h': h}


    def forward(self, g, prev_h, emb_rel):
        self.rel_emb = emb_rel

        if self.self_loop:
            loop_message = torch.mm(g.ndata['h'], self.loop_weight)
        if len(prev_h) != 0 and self.skip_connect:
            skip_weight = F.sigmoid(torch.mm(prev_h, self.skip_connect_weight) + self.skip_connect_bias) 
        g.apply_edges(self.edge_attention) 

        self.propagate(g)
        node_repr = g.ndata['h']

        if len(prev_h) != 0 and self.skip_connect:  
            if self.self_loop:
                node_repr = node_repr + loop_message
            node_repr = skip_weight * node_repr + (1 - skip_weight) * prev_h
        else:
            if self.self_loop:
                node_repr = node_repr + loop_message

        if self.activation:
            node_repr = self.activation(node_repr)
        if self.dropout is not None:
            node_repr = self.dropout(node_repr)
        g.ndata['h'] = node_repr
        return node_repr
    
class CompGCNLayer2(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, comp, num_bases=-1,  bias=None,
                 activation=None, self_loop=False, dropout=0.0, skip_connect=False, rel_emb=None):
        super(CompGCNLayer2, self).__init__()

        self.in_feat = in_feat
        self.out_feat = out_feat
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.num_rels = num_rels
        self.rel_emb = None
        self.skip_connect = skip_connect
        self.ob = None
        self.sub = None
        self.comp = comp

        # WL
        self.weight_neighbor = nn.Parameter(torch.Tensor(self.in_feat, self.out_feat))
        nn.init.xavier_uniform_(self.weight_neighbor, gain=nn.init.calculate_gain('relu'))

        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
            self.evolve_loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.evolve_loop_weight, gain=nn.init.calculate_gain('relu'))

        if self.skip_connect:
            self.skip_connect_weight = nn.Parameter(torch.Tensor(out_feat, out_feat))   # 和self-loop不一样，是跨层的计算
            nn.init.xavier_uniform_(self.skip_connect_weight,gain=nn.init.calculate_gain('relu'))
            self.skip_connect_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.skip_connect_bias)  # 初始化设置为0

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def propagate(self, g):
        g.update_all(lambda x: self.msg_func(x), fn.sum(msg='msg', out='h'), self.apply_func)

    def forward(self, g, prev_h, emb_rel):
        self.rel_emb = emb_rel
        # self.sub = sub
        # self.ob = ob
        if self.self_loop:
            loop_message = torch.mm(g.ndata['h'], self.loop_weight)

        if len(prev_h) != 0 and self.skip_connect:
            skip_weight = F.sigmoid(torch.mm(prev_h, self.skip_connect_weight) + self.skip_connect_bias)     
        self.propagate(g)
        node_repr = g.ndata['h']

        if len(prev_h) != 0 and self.skip_connect:  
            if self.self_loop:
                node_repr = node_repr + loop_message
            node_repr = skip_weight * node_repr + (1 - skip_weight) * prev_h
        else:
            if self.self_loop:
                node_repr = node_repr + loop_message

        if self.activation:
            node_repr = self.activation(node_repr)
        if self.dropout is not None:
            node_repr = self.dropout(node_repr)
        g.ndata['h'] = node_repr
        return node_repr

    def msg_func(self, edges):

        relation = self.rel_emb.index_select(0, edges.data['type']).view(-1, self.out_feat)
        edge_type = edges.data['type']
        edge_num = edge_type.shape[0]
        node = edges.src['h'].view(-1, self.out_feat)

        if self.comp == "sub":
            msg = node + relation
        elif self.comp == "mult":
            msg = node * relation

        msg = torch.mm(msg, self.weight_neighbor)
        return {'msg': msg}

    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['norm']}