import csv
from datetime import datetime
import argparse
import os
import sys
import time
import pickle
import dgl
import numpy as np
import torch
from tqdm import tqdm
import random
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset

sys.path.append(".")
from rgcn import utils
from rgcn.utils import build_sub_graph, build_graph
from src.rrgcn import RecurrentRGCN
from collections import defaultdict
from rgcn.knowledge_graph import _read_triplets_as_list
import pandas as pd
import warnings
import logging
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')


def update_dict(subg_arr, s_to_sro, sr_to_sro, sro_to_fre, num_rels):
    inverse_subg = subg_arr[:, [2, 1, 0]]
    inverse_subg[:, 1] += num_rels
    subg_triples = np.concatenate([subg_arr, inverse_subg])
    for src, rel, dst in subg_triples:
        s_to_sro[src].add((src, rel, dst))
        sr_to_sro[(src, rel)].add(dst)


def e2r(triplets, num_rels):
    src, rel, _ = triplets.transpose()
    uniq_e = np.unique(src)
    e_to_r = defaultdict(set)
    for s, r in zip(src, rel):
        e_to_r[s].add(r)
    r_len, r_idx, idx = [], [], 0
    for e in uniq_e:
        r_len.append((idx, idx + len(e_to_r[e])))
        r_idx.extend(e_to_r[e])
        idx += len(e_to_r[e])
    return [torch.from_numpy(np.array(uniq_e)).long().cuda(),
            torch.from_numpy(np.array(r_len)).long().cuda(),
            torch.from_numpy(np.array(r_idx)).long().cuda()]


def get_sample_from_history_graph3(subg_arr, sr_to_sro, triples,
                                   num_nodes, num_rels, use_cuda, gpu):
    inverse_triples = triples[:, [2, 1, 0]]
    inverse_triples[:, 1] += num_rels
    src_set = set(triples[:, 0])
    dst_set = set(triples[:, 0])
    er_list = list({(h, r) for h, r in triples[:, :2]})
    er_list_inv = list({(h, r) for h, r in inverse_triples[:, :2]})
    inverse_subg = subg_arr[:, [2, 1, 0]]
    inverse_subg[:, 1] += num_rels
    subg_triples = np.concatenate([subg_arr, inverse_subg])
    df = pd.DataFrame(subg_triples, columns=['src', 'rel', 'dst'])
    subg_df = df.groupby(df.columns.tolist()).size().reset_index().rename(columns={0: 'freq'})

    df_dic = pd.DataFrame({'sr': list(sr_to_sro.keys()),
                           'dst': list(sr_to_sro.values())})

    dst_df = df_dic.query('sr in @er_list')
    two_ent = set().union(*dst_df['dst'].values) if not dst_df.empty else set()
    result = subg_df.query('src in @list(src_set | two_ent)')
    dst_df_inv = df_dic.query('sr in @er_list_inv')
    two_ent_inv = set().union(*dst_df_inv['dst'].values) if not dst_df_inv.empty else set()
    result_inv = subg_df.query('src in @list(dst_set | two_ent_inv)')

    q_tri = result.to_numpy()
    q_tri_inv = result_inv.to_numpy()
    q_tri_all = np.concatenate([q_tri, q_tri_inv], axis=0)
    his_sub = build_graph(num_nodes, num_rels, q_tri, use_cuda, gpu)
    his_sub_inv = build_graph(num_nodes, num_rels, q_tri_inv, use_cuda, gpu)
    his_sub_all = build_graph(num_nodes, num_rels, q_tri_all, use_cuda, gpu)
    return his_sub_all, his_sub_inv


def test(model, history_list, test_list, num_rels, num_nodes, use_cuda,
         all_ans_list, all_ans_r_list, model_name, static_graph, mode,
         sr_to_sro):
    ranks_raw, ranks_filter = [], []
    if mode == "test":
        checkpoint = torch.load(model_name,
                                map_location=torch.device('cuda' if use_cuda else 'cpu'))
        model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    input_list = [s for s in history_list[-args.test_history_len:]]
    subg_arr = np.concatenate(history_list)

    for time_idx, test_snap in enumerate(tqdm(test_list, disable=True)):
        history_glist = [build_sub_graph(num_nodes, num_rels, g, use_cuda, args.gpu)
                         for g in input_list]
        inverse_triples = test_snap[:, [2, 1, 0]]
        inverse_triples[:, 1] += num_rels
        que_pair = e2r(test_snap, num_rels)
        que_pair_inv = e2r(inverse_triples, num_rels)
        triples_all = np.concatenate([test_snap, inverse_triples], 0)
        que_pair_all = que_pair + que_pair_inv

        sub_snap, sub_snap_inv = get_sample_from_history_graph3(
            subg_arr, sr_to_sro, test_snap, num_nodes, num_rels,
            use_cuda, args.gpu)
        test_triples_input_all = torch.as_tensor(triples_all).long().to(
            'cuda' if use_cuda else 'cpu')
        test_triples, final_score = model.predict(
            que_pair_all, sub_snap, time_idx, history_glist,
            num_rels, static_graph, test_triples_input_all, use_cuda)
        _, _, rank_raw, rank_filter = utils.get_total_rank(
            test_triples, final_score, all_ans_list[time_idx],
            eval_bz=1000, rel_predict=0)

        ranks_raw.append(rank_raw)
        ranks_filter.append(rank_filter)
        if args.multi_step and not args.relation_evaluation:
            predicted_snap = utils.construct_snap(test_triples, num_nodes,
                                                  num_rels, final_score, args.topk)
            if len(predicted_snap):
                input_list.pop(0)
                input_list.append(predicted_snap)
        else:
            input_list.pop(0)
            input_list.append(test_snap)
    mrr_raw, _ = utils.stat_ranks(ranks_raw, "raw")
    mrr_filter, _ = utils.stat_ranks(ranks_filter, "filter")
    return mrr_raw, mrr_filter


def run_experiment(args, n_hidden=None, n_layers=None, dropout=None, n_bases=None):
    if n_hidden: args.n_hidden = n_hidden
    if n_layers: args.n_layers = n_layers
    if dropout: args.dropout = dropout
    if n_bases: args.n_bases = n_bases
    data = utils.load_data(args.dataset)
    train_list = utils.split_by_time(data.train)
    valid_list = utils.split_by_time(data.valid)
    test_list = utils.split_by_time(data.test)
    train_list = random.sample(train_list, int(0.9 * len(train_list)))
    num_nodes, num_rels = data.num_nodes, data.num_rels

    pkl_path = os.path.join(BASE_DIR, 'data', args.dataset, 'his_graph.pkl')
    with open(pkl_path, 'rb') as f:
        _pkl = pickle.load(f)
    his_graph_for_list = _pkl['his_graph_for']
    his_graph_inv_list = _pkl['his_graph_inv']
    sr_to_sro = _pkl['sr_to_sro']

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    static_graph = None

    model = RecurrentRGCN(
        args.decoder, args.encoder, num_nodes, num_rels,
        0, 0, args.n_hidden, args.opn,
        sequence_len=args.train_history_len, num_bases=args.n_bases,
        num_basis=args.n_basis, num_hidden_layers=args.n_layers,
        dropout=args.dropout, self_loop=args.self_loop,
        skip_connect=args.skip_connect, layer_norm=args.layer_norm,
        input_dropout=args.input_dropout, hidden_dropout=args.hidden_dropout,
        feat_dropout=args.feat_dropout, aggregation=args.aggregation,
        weight=args.weight, pre_weight=args.pre_weight,
        discount=args.discount, angle=args.angle,
        use_static=args.add_static_graph, pre_type=args.pre_type,
        use_cl=args.use_cl, temperature=args.temperature,
        entity_prediction=args.entity_prediction,
        relation_prediction=args.relation_prediction,
        use_cuda=use_cuda, gpu=args.gpu, analysis=args.run_analysis,
        num_clusters=args.num_clusters)

    if use_cuda:
        torch.cuda.set_device(args.gpu)
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr, weight_decay=1e-5)

    for epoch in range(args.n_epochs):
        model.train()
        idx = list(range(len(train_list)))

        for train_sample_num in tqdm(idx, disable=True):
            if train_sample_num == 0: continue
            output = train_list[train_sample_num:train_sample_num + 1]
            if train_sample_num - args.train_history_len < 0:
                input_list = train_list[:train_sample_num]
            else:
                start = train_sample_num - args.train_history_len
                input_list = train_list[start:train_sample_num]

            subgraph_arr      = his_graph_for_list[train_sample_num]
            subgraph_arr_inv  = his_graph_inv_list[train_sample_num]
            subg_snap = build_graph(num_nodes, num_rels, subgraph_arr,
                                    use_cuda, args.gpu)
            subg_snap_inv = build_graph(num_nodes, num_rels, subgraph_arr_inv,
                                        use_cuda, args.gpu)
            inverse_triples = output[0][:, [2, 1, 0]]
            inverse_triples[:, 1] += num_rels
            que_pair = e2r(output[0], num_rels)
            que_pair_inv = e2r(inverse_triples, num_rels)
            history_glist = [build_sub_graph(num_nodes, num_rels, snap,
                                             use_cuda, args.gpu) for snap in input_list]
            triples = torch.as_tensor(output[0]).long().cuda()
            inverse_triples = torch.as_tensor(inverse_triples).long().cuda()

            for id_ in range(2):
                if id_ % 2 == 0:
                    loss_e, loss_r, loss_s, loss_cl, loss_pcl = model.get_loss(
                        que_pair, subg_snap, train_sample_num, history_glist,
                        triples, static_graph, use_cuda)
                else:
                    loss_e, loss_r, loss_s, loss_cl, loss_pcl = model.get_loss(
                        que_pair_inv, subg_snap_inv, train_sample_num,
                        history_glist, inverse_triples,
                        static_graph, use_cuda)
                loss = loss_e + loss_s + args.lambda_cl * loss_cl + args.lambda_pcl * loss_pcl
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
                optimizer.step()
                optimizer.zero_grad()
            torch.cuda.empty_cache()

        if epoch and epoch % args.evaluate_every == 0:
            test(model, train_list, valid_list, num_rels, num_nodes,
                 use_cuda, None, None, '', static_graph,
                 mode="train", sr_to_sro=sr_to_sro)
            torch.cuda.empty_cache()

    mrr_raw, mrr_filter = test(model, train_list + valid_list, test_list,
                               num_rels, num_nodes, use_cuda,
                               None, None, '', static_graph,
                               mode="test", sr_to_sro=sr_to_sro)
    return mrr_raw, mrr_filter


if __name__ == '__main__':
    parser.add_argument("--opn", type=str, default="sub")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lambda-cl", type=float, default=0.8)
    parser.add_argument("--decoder", type=str, default="convtranse")
    parser.add_argument("--test", action='store_true', default=False)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--evaluate-every", type=int, default=1)
    parser.add_argument("--skip-connect", action='store_true', default=False)
    parser.add_argument("--n-basis", type=int, default=100)
    parser.add_argument("--relation-evaluation", action='store_true', default=False)
    parser.add_argument("--angle", type=int, default=10)
    parser.add_argument("--hidden-dropout", type=float, default=0.2)
    parser.add_argument("--input-dropout", type=float, default=0.2)
    parser.add_argument("--lambda-pcl", type=float, default=0.2)
    parser.add_argument("--aggregation", type=str, default="none")
    parser.add_argument("--relation-prediction", action='store_true', default=False)
    parser.add_argument("--split_by_relation", action='store_true', default=False)
    parser.add_argument("--self-loop", action='store_true', default=True)
    parser.add_argument("--train-history-len", type=int, default=7)
    parser.add_argument("--dilate-len", type=int, default=1)
    parser.add_argument("--n-epochs", type=int, default=20)
    parser.add_argument("--feat-dropout", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.03)
    parser.add_argument("--run-analysis", action='store_true', default=False)
    parser.add_argument("--test-history-len", type=int, default=20)
    parser.add_argument("--entity-prediction", action='store_true', default=True)
    parser.add_argument("--n-hidden", type=int, default=200)
    parser.add_argument("--n-bases", type=int, default=100)
    parser.add_argument("--run-statistic", action='store_true', default=False)
    parser.add_argument("--add-static-graph", action='store_true', default=True)
    parser.add_argument("--pre-weight", type=float, default=0.9)
    parser.add_argument("--encoder", type=str, default="uvrgcn")
    parser.add_argument("--dataset", "-d", type=str, default="ICEWS14")
    parser.add_argument("--use-cl", action='store_true', default=True)
    parser.add_argument("--grad-norm", type=float, default=1.0)
    parser.add_argument("--add-rel-word", action='store_true', default=False)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--num-clusters", type=int, default=20)
    parser.add_argument("--discount", type=float, default=1)
    parser.add_argument("--weight", type=float, default=0.5)
    parser.add_argument("--layer-norm", action='store_true', default=True)
    parser.add_argument("--pre-type", type=str, default="all")

    args = parser.parse_args()


    run_experiment(args)
