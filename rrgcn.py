import os, math, json, logging, torch, numpy as np
import torch.nn as nn, torch.nn.functional as F
from torch.nn import MultiheadAttention
from collections import defaultdict
from sklearn.cluster import MiniBatchKMeans
import torch.utils.checkpoint as checkpoint

from rgcn.layers import (UnionRGCNLayer, RGCNBlockLayer, RGAT,
                         UnionRGCNLayer2, UnionRGATLayer, CompGCNLayer)
from src.model import BaseRGCN
from src.decoder import ConvTransE, ConvTransR


_EPS = 1e-8


def _guard_tensor(t, name='', dim=None):
    if not isinstance(t, torch.Tensor):
        raise TypeError(f'{name or "tensor"} must be torch.Tensor')
    if dim is not None and t.dim() < dim:
        raise ValueError(f'{name or "tensor"} requires dim>={dim}')
    if torch.isnan(t).any() or torch.isinf(t).any():
        t.clamp_(min=-1e4, max=1e4)

def _norm(x):  # safe l2 normalize
    return F.normalize(x, p=2, dim=1) if x.dim() > 1 else x



class AttentionMechanism(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = .1):
        super().__init__()
        self.attn = MultiheadAttention(embed_dim, num_heads, dropout=dropout)

    def forward(self, his_emb: torch.Tensor, history_emb: torch.Tensor):
        _guard_tensor(his_emb, 'his_emb', 2)
        _guard_tensor(history_emb, 'history_emb', 2)

        q = his_emb.unsqueeze(1).transpose(0, 1)       # [L,1,D]
        k = history_emb.unsqueeze(1).transpose(0, 1)
        v = k
        
        def _f(Q, K, V):
            return self.attn(Q, K, V)[0]
        try:
            out = checkpoint.checkpoint(_f, q, k, v).squeeze(1)
        except RuntimeError:                           # OOM 回退
            out = self.attn(q, k, v)[0].squeeze(1)
        return out

    @staticmethod
    def concat_embeddings(his_emb: torch.Tensor, history_emb: torch.Tensor):
        assert his_emb.shape == history_emb.shape
        return torch.cat((his_emb, history_emb), dim=1)



class MLPLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.l1 = nn.Linear(in_dim, out_dim)
        self.l2 = nn.Linear(out_dim, out_dim)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        _guard_tensor(x, 'mlp_in', 2)
        return self.act(_norm(self.l2(self.act(_norm(self.l1(x))))))




class RGCNCell(BaseRGCN):
    def build_hidden_layer(self, idx: int):
        act = F.rrelu
        sc_flag = self.skip_connect and idx != 0
        if self.encoder_name == "uvrgcn":
            return UnionRGCNLayer(self.h_dim, self.h_dim, self.num_rels,
                                  self.num_bases, activation=act,
                                  self_loop=self.self_loop,
                                  dropout=self.dropout,
                                  skip_connect=sc_flag,
                                  rel_emb=self.rel_emb)
        if self.encoder_name == "kbat":
            return UnionRGATLayer(self.h_dim, self.h_dim, self.num_rels,
                                  self.num_bases, rel_emb=self.rel_emb)
        if self.encoder_name == "compgcn":
            return CompGCNLayer(self.h_dim, self.h_dim, self.num_rels,
                                self.opn, self.num_bases, activation=act,
                                self_loop=self.self_loop,
                                dropout=self.dropout,
                                skip_connect=sc_flag, rel_emb=self.rel_emb)
        raise NotImplementedError

    def forward(self, g, ent_emb, rel_emb):
        node_id = g.ndata['id'].squeeze()
        g.ndata['h'] = ent_emb[node_id]
        if self.encoder_name in ["uvrgcn", "kbat", "compgcn"]:
            for i, lyr in enumerate(self.layers):
                lyr(g, [], rel_emb[i])
            return g.ndata.pop('h')
        prev = []
        for lyr in self.layers:
            prev = lyr(g, prev)
        return g.ndata.pop('h')


class RGCNCell2(BaseRGCN):
    def build_hidden_layer(self, idx):
        sc_flag = self.skip_connect and idx != 0
        return UnionRGCNLayer2(self.h_dim, self.h_dim, self.num_rels,
                               self.num_bases, activation=F.rrelu,
                               dropout=self.dropout, self_loop=self.self_loop,
                               skip_connect=sc_flag, rel_emb=self.rel_emb)

    def forward(self, g, ent_emb, rel_emb):
        node_id = g.ndata['id'].squeeze()
        g.ndata['h'] = ent_emb[node_id]
        for i, lyr in enumerate(self.layers):
            lyr(g, [], rel_emb[i])
        return g.ndata.pop('h')




class RecurrentRGCN(nn.Module):
    def __init__(self, decoder_name, encoder_name, num_ents, num_rels,
                 num_static_rels, num_words, h_dim, opn, num_clusters,
                 sequence_len, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False,
                 skip_connect=False, layer_norm=False, input_dropout=0,
                 hidden_dropout=0, feat_dropout=0, aggregation='cat',
                 weight=1., pre_weight=.7, discount=0, angle=0,
                 use_static=False, pre_type='short', log_dir="logs",
                 use_cl=False, temperature=.007, margin=.5,
                 entity_prediction=False, relation_prediction=False,
                 use_cuda=False, gpu=0, analysis=False):
        super().__init__()

        os.makedirs(log_dir, exist_ok=True)
        self.logger = logging.getLogger("RecurrentRGCN")
        if not self.logger.handlers:
            fh = logging.FileHandler(os.path.join(log_dir, "training.log"))
            fh.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
            self.logger.addHandler(fh)
        self.logger.setLevel(logging.WARNING)

        self.decoder_name = decoder_name
        self.encoder_name = encoder_name
        self.num_rels, self.num_ents = num_rels, num_ents
        self.opn = opn
        self.num_clusters = num_clusters
        self.num_static_rels, self.num_words = num_static_rels, num_words
        self.sequence_len, self.h_dim, self.layer_norm = sequence_len, h_dim, layer_norm
        self.aggregation, self.weight, self.pre_weight = aggregation, weight, pre_weight
        self.discount, self.use_static = discount, use_static
        self.margin, self.pre_type, self.use_cl = margin, pre_type, use_cl
        self.temp, self.angle = temperature, angle
        self.relation_prediction, self.entity_prediction = relation_prediction, entity_prediction
        self.gpu = gpu

        self.emb_rel = nn.Parameter(torch.empty(num_rels * 2, h_dim))
        self.dynamic_emb = nn.Parameter(torch.empty(num_ents, h_dim))
        nn.init.xavier_normal_(self.emb_rel)
        nn.init.normal_(self.dynamic_emb)

        if self.use_static:
            self.words_emb = nn.Parameter(torch.empty(num_words, h_dim))
            nn.init.xavier_normal_(self.words_emb)
            self.stat_layer = RGCNBlockLayer(h_dim, h_dim, num_static_rels * 2,
                                             num_bases, activation=F.rrelu,
                                             dropout=dropout)

        self.projection_model = MLPLinear(h_dim, h_dim)
        self.cross_att = AttentionMechanism(h_dim, 8, 0.3)
        self.weight_mlp = nn.Sequential(
            nn.Linear(2 * h_dim, h_dim), nn.Tanh(),
            nn.Linear(h_dim, 1), nn.Sigmoid()).to(gpu)

        self.time_gate_weight = nn.Parameter(torch.empty(h_dim, h_dim))
        self.time_gate_bias = nn.Parameter(torch.zeros(h_dim))
        nn.init.xavier_uniform_(self.time_gate_weight)

        self.entity_cell = nn.GRUCell(h_dim, h_dim)
        self.relation_cell = nn.GRUCell(h_dim, h_dim)

        self.rgcn = RGCNCell(num_ents, h_dim, h_dim, num_rels * 2,
                             num_bases, num_basis, num_hidden_layers,
                             dropout, self_loop, skip_connect,
                             encoder_name, opn, self.emb_rel, use_cuda,
                             analysis)
        self.his_rgcn_layer = RGCNCell2(num_ents, h_dim, h_dim,
                                        num_rels * 2, num_bases, num_basis,
                                        num_hidden_layers, dropout,
                                        self_loop, skip_connect,
                                        encoder_name, opn, self.emb_rel,
                                        use_cuda, analysis)

        if decoder_name == "convtranse":
            self.decoder_ob = ConvTransE(num_ents, h_dim,
                                         input_dropout, hidden_dropout,
                                         feat_dropout)
            self.rdecoder = ConvTransR(num_rels, h_dim,
                                       input_dropout, hidden_dropout,
                                       feat_dropout)
        else:
            raise NotImplementedError

        self.loss_r = nn.CrossEntropyLoss()
        self.loss_e = nn.NLLLoss()

    def get_emb(self, static_graph):
        if self.use_static:
            static_graph = static_graph.to(self.gpu)
            static_graph.ndata['h'] = torch.cat((self.dynamic_emb, self.words_emb), 0)
            self.stat_layer(static_graph, [])
            static_emb = static_graph.ndata.pop('h')[:self.num_ents]
            static_emb = _norm(static_emb) if self.layer_norm else static_emb
            return static_emb
        return _norm(self.dynamic_emb) if self.layer_norm else self.dynamic_emb

    def forward(self, sub_graph, T_idx, query_mask, g_list, static_graph, use_cuda):
        if self.use_static:
            static_graph = static_graph.to(self.gpu)
            static_graph.ndata['h'] = torch.cat((self.dynamic_emb, self.words_emb), 0)
            self.statci_layer(static_graph, [])
            static_emb = static_graph.ndata.pop('h')[:self.num_ents]
            static_emb = F.normalize(static_emb) if self.layer_norm else static_emb
            self.h = static_emb
        else:
            self.h = F.normalize(self.dynamic_emb) if self.layer_norm else self.dynamic_emb
            static_emb = None

        self.his_ent, _ = self.all_GCN(self.h, sub_graph, use_cuda)
        his_r_emb = F.normalize(self.emb_rel)
        his_att = F.softmax(self.w5(query_mask + self.his_ent), dim=1)
        his_emb = F.normalize(his_att * self.his_ent)

        history_embs, att_embs, his_temp_embs, his_rel_embs = [], [], [], []

        if self.pre_type == "all":
            h, hr = self.dynamic_emb, self.emb_rel

            def process_step(h_, hr_, g_, t2_, qmask_):
                h_t = torch.cos(self.weight_t2 * t2_ + self.bias_t2).repeat(self.num_ents, 1).to(self.gpu)
                h_tw = self.w4(torch.cat((h_, h_t), 1))
                temp_e = h_tw[g_.r_to_e.long().to(self.gpu)]
                x_input = torch.zeros(self.num_rels * 2, self.h_dim, device=self.gpu)
                for span, r_idx in zip(g_.r_len, g_.uniq_r):
                    x_mean = temp_e[span[0]:span[1]].mean(0, keepdim=True)
                    x_input[r_idx] = x_mean
                x_input = hr_ + x_input
                cur_h = self.rgcn.forward(g_, h_tw, [self.emb_rel, self.emb_rel])
                cur_h = F.normalize(cur_h) if self.layer_norm else cur_h
                att_e = F.softmax(self.w2(qmask_ + cur_h), 1)
                att_emb = att_e * cur_h
                h_new = self.entity_cell(cur_h, h_)
                h_new = F.normalize(h_new) if self.layer_norm else h_new
                tw = torch.sigmoid(torch.mm(x_input, self.time_gate_weight) + self.time_gate_bias)
                hr_new = F.normalize(tw * x_input + (1 - tw) * hr_) if self.layer_norm else tw * x_input + (1 - tw) * hr_
                return h_new, hr_new, att_emb

            for i, g in enumerate(g_list):
                t2 = len(g_list) - i + 1
                g = g.to(self.gpu)
                try:
                    self.h, self.hr, att_e = checkpoint.checkpoint(
                        process_step, h, hr, g, torch.tensor(t2, device=self.gpu), query_mask)
                except RuntimeError:  
                    self.h, self.hr, att_e = process_step(h, hr, g, torch.tensor(t2, device=self.gpu), query_mask)
                history_embs.append(self.h)
                his_rel_embs.append(self.hr)
                his_temp_embs.append(self.h)
                att_embs.append(att_e.unsqueeze(0))

            att_ent = F.normalize(torch.mean(torch.cat(att_embs, 0), 0))
            history_emb = F.normalize(att_ent + history_embs[-1]) if self.layer_norm else att_ent + history_embs[-1]
            gl_output = self.cross_att(his_emb, history_emb)
            lg_output = self.cross_att(history_emb, his_emb)
        else:
            self.hr, history_emb, gl_output, lg_output = None, None, None, None

        return (history_emb, static_emb, self.hr, his_emb, his_r_emb,
                his_temp_embs, his_rel_embs, gl_output, lg_output)


    def predict(self, que_pair, sub_graph, T_id, test_graph,
                num_rels, static_graph, test_triplets, use_cuda):

        with torch.no_grad():
            all_triples = test_triplets
            uniq_e, r_len, r_idx = que_pair
            temp_r = self.emb_rel[r_idx]

            e_input = torch.zeros(self.num_ents, self.h_dim,
                                device='cuda' if use_cuda else 'cpu')
            for span, e_idx in zip(r_len, uniq_e):
                e_input[e_idx] = temp_r[span[0]:span[1]].mean(0)

            query_mask = torch.zeros((self.num_ents, self.h_dim),
                                    device=self.gpu if use_cuda else 'cpu')
            e1_emb = self.dynamic_emb[uniq_e]
            rel_emb = e_input[uniq_e]
            query_mask[uniq_e] = self.w1(torch.cat((e1_emb, rel_emb), 1))

            (embedding, _, r_emb, his_emb, _, _, _,
            gl_output, lg_output) = self.forward(sub_graph, T_id,
                                                query_mask, test_graph,
                                                static_graph, use_cuda)

            if self.pre_type == "all":
                scores_ob, _ = self.decoder_ob.forward(
                    embedding, gl_output, lg_output, r_emb,
                    all_triples, his_emb, self.pre_weight, self.pre_type)
                score_seq = F.softmax(scores_ob, 1)
                score_en = score_seq + _EPS                 
            else:
                raise RuntimeError("pre_type must be 'all' for predict")

            scores_en = torch.log(score_en)
            return all_triples, scores_en
            
    def get_loss(self, que_pair, sub_graph, T_idx, g_list,
                triples, static_graph, use_cuda):

        dev = self.gpu if use_cuda else 'cpu'
        zeros = lambda: torch.zeros(1, device=dev)

        loss_ent, loss_cl, loss_pcl, loss_rel, loss_static = map(lambda _: zeros(), range(5))
        all_triples = triples

        uniq_e, r_len, r_idx = que_pair
        _guard_tensor(self.emb_rel, 'emb_rel', 2)
        temp_r = self.emb_rel[r_idx]

        e_input = torch.zeros(self.num_ents, self.h_dim, device=dev)
        for span, e_idx in zip(r_len, uniq_e):
            e_input[e_idx] = temp_r[span[0]:span[1]].mean(0)

        query_mask = torch.zeros((self.num_ents, self.h_dim), device=dev)
        q_t = torch.cos(self.weight_t2 * 0 + self.bias_t2).repeat(self.num_ents, 1)
        qe_emb = self.w4(torch.cat((self.dynamic_emb, q_t.to(dev)), 1))
        query_mask[uniq_e] = self.w1(torch.cat((qe_emb[uniq_e], e_input[uniq_e]), 1))

        (embedding, _, r_emb, his_emb, his_r_emb,
        his_temp_embs, his_rel_embs, gl_out, lg_out) = self.forward(
            sub_graph, T_idx, query_mask, g_list, static_graph, use_cuda)

        if self.pre_type == "all":
            scores_ob, _ = self.decoder_ob.forward(
                embedding, gl_out, lg_out, r_emb, all_triples,
                his_emb, self.pre_weight, self.pre_type)
            score_en = F.softmax(scores_ob, 1) + _EPS
        else:
            raise RuntimeError("pre_type must be 'all' during training")

        scores_log = torch.log(score_en)
        loss_ent += F.nll_loss(scores_log, triples[:, 2])

        if self.relation_prediction:
            logits_r = self.rdecoder.forward(
                embedding, r_emb, all_triples, mode="train").view(-1, 2 * self.num_rels)
            loss_rel += self.loss_r(logits_r, all_triples[:, 1])

        if self.use_cl and self.pre_type == "all":
            x1_l, x2_l = [], []
            for idx_t, evo_emb in enumerate(his_temp_embs):
                q1 = torch.cat((self.his_ent[all_triples[:, 0]],
                                his_r_emb[all_triples[:, 1]]), 1)
                q2 = torch.cat((evo_emb[all_triples[:, 0]],
                                his_rel_embs[idx_t][all_triples[:, 1]]), 1)
                x1, x2 = self.w_cl(q1), self.w_cl(q2)
                x1_l.append(x1); x2_l.append(x2)
                loss_cl += self.get_loss_conv(x1, x2)

            x1_cat = torch.cat(x1_l)
            x2_cat = torch.cat(x2_l)
            event_emb = (x1_cat + x2_cat) * 0.5
            cluster = self.perform_kmeans(event_emb)
            cluster_rep = cluster.repeat_interleave(2)
            emb_full = torch.cat((x1_cat, x2_cat), 0)
            loss_pcl += self.get_loss_conv_category(emb_full, cluster_rep, margin=self.margin)

        return loss_ent, loss_rel, loss_static, loss_cl, loss_pcl


    def all_GCN(self, ent_emb, sub_graph, use_cuda):
        sub_graph = sub_graph.to(self.gpu)
        sub_graph.ndata['h'] = ent_emb
        his_emb = self.his_rgcn_layer.forward(
            sub_graph, ent_emb, [self.emb_rel, self.emb_rel])
        idx_mask = (sub_graph.in_degrees(range(sub_graph.num_nodes())) > 0)
        subg_index = torch.masked_select(torch.arange(sub_graph.num_nodes(),
                                    device=self.gpu, dtype=torch.long), idx_mask)
        return _norm(his_emb), subg_index


    def perform_kmeans(self, embeddings):
        with torch.no_grad():
            emb_np = embeddings.detach().cpu().numpy()
            n_samp = emb_np.shape[0]
            n_c = min(self.num_clusters, n_samp)
            if n_c < 2:
                return torch.arange(n_samp, device=embeddings.device)
            kmeans = MiniBatchKMeans(n_clusters=n_c, random_state=0, batch_size=1024)
            labels = kmeans.fit_predict(emb_np)
            return torch.tensor(labels, device=embeddings.device, dtype=torch.long)


    def get_loss_conv(self, ent1_emb, ent2_emb):
        z1 = _norm(self.projection_model(ent1_emb))
        z2 = _norm(self.projection_model(ent2_emb))
        sim = torch.mm(z1, z2.T) / (self.temp + _EPS)
        H = torch.cat((z1.unsqueeze(1).expand(-1, z2.size(0), -1),
                    z2.unsqueeze(0).expand(z1.size(0), -1, -1)), 2)
        gamma = self.weight_mlp(H.view(-1, 2 * self.embedding_dim)).squeeze().view(z1.size(0), z2.size(0)) + _EPS
        log_prob = sim + gamma.log() - torch.logsumexp(sim + gamma.log(), 1, keepdim=True)
        return (-torch.diagonal(log_prob)).mean()


    def get_loss_conv_category(self, embeddings, cluster_labels, margin=0.5):
        unique = torch.unique(cluster_labels)
        centroid_list = []
        for lab in unique:
            m = (cluster_labels == lab)
            if m.sum() == 0:
                continue
            centroid_list.append(embeddings[m].mean(0))
        if not centroid_list:
            return torch.zeros(1, device=embeddings.device)

        centroids = torch.stack(centroid_list)
        assign = centroids[cluster_labels]
        pos_sim = F.cosine_similarity(embeddings, assign)
        neg_sim = F.cosine_similarity(embeddings.unsqueeze(1), centroids.unsqueeze(0), 2)
        mask = torch.zeros_like(neg_sim, dtype=torch.bool)
        mask[torch.arange(embeddings.size(0)), cluster_labels] = True
        neg_sim[mask] = float('-inf')
        neg_max = neg_sim.max(1)[0]
        return F.relu(margin - pos_sim + neg_max).mean()