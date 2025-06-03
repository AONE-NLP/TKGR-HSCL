
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import torch
import dgl
from tqdm import tqdm
import rgcn.knowledge_graph as knwlgrh  # keep for API compatibility

def _assert_tensor(t: torch.Tensor, dim: int, dtype, name: str):
    if not isinstance(t, torch.Tensor):
        raise TypeError(f'{name} must be torch.Tensor')
    if t.dim() != dim:
        raise ValueError(f'{name} must be {dim}‑D (got {t.dim()})')
    if t.dtype != dtype:
        raise TypeError(f'{name} must be {dtype}, got {t.dtype}')


def _assert_ndarray(a: np.ndarray, dim: int, dtype, name: str):
    if not isinstance(a, np.ndarray):
        raise TypeError(f'{name} must be np.ndarray')
    if a.ndim != dim:
        raise ValueError(f'{name} must be {dim}‑D (got {a.ndim})')
    if a.dtype != dtype:
        raise TypeError(f'{name} must be {dtype}, got {a.dtype}')


def _safe_softmax_numpy(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    x = np.clip(x, -50.0, 50.0)
    e = np.exp(x)
    return e / np.sum(e)

def sort_and_rank(score: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    _assert_tensor(score, 2, torch.float32, 'score')
    _assert_tensor(target, 1, torch.long, 'target')
    idx_sorted = torch.argsort(score, dim=1, descending=True)
    idx_pos = (idx_sorted == target.view(-1, 1)).nonzero()[:, 1]
    return idx_pos.view(-1)


def sort_and_rank_time_filter(batch_a, batch_r, score, target, total_triplets):
    return sort_and_rank(score, target)


def sort_and_rank_filter(batch_a, batch_r, score, target, all_ans):
    _assert_tensor(score, 2, torch.float32, 'score')
    _assert_tensor(target, 1, torch.long, 'target')
    for i in range(len(batch_a)):
        ans = target[i]
        mask_idx = list(all_ans[batch_a[i].item()][batch_r[i].item()])
        ref_val = score[i, ans]
        score[i, mask_idx] = 0
        score[i, ans] = ref_val
    idx_sorted = torch.argsort(score, dim=1, descending=True)
    idx_pos = (idx_sorted == target.view(-1, 1)).nonzero()[:, 1]
    return idx_pos.view(-1)

def _generic_filter(test_triples, score, all_ans, remover):
    if all_ans is None:
        return score
    _assert_tensor(score, 2, torch.float32, 'score')
    triples_cpu = test_triples.cpu()
    score = score.clone()                          
    for row, (h, r, t) in enumerate(triples_cpu):
        ban = remover(h.item(), r.item(), t.item(), all_ans)
        if ban:
            score[row, torch.LongTensor(ban)] = -1e7
    return score


def filter_score(test_triples, score, all_ans):
    return _generic_filter(
        test_triples, score, all_ans,
        lambda h, r, t, a: [x for x in a[h][r] if x != t] if h in a and r in a[h] else []
    )


def filter_score_r(test_triples, score, all_ans):
    return _generic_filter(
        test_triples, score, all_ans,
        lambda h, r, t, a: [x for x in a[h][t] if x != r] if h in a and t in a[h] else []
    )

def r2e(triplets: np.ndarray, num_rels: int):
    _assert_ndarray(triplets, 2, np.int64, 'triplets')
    src, rel, dst = triplets.transpose()
    uniq_r = np.unique(rel)
    uniq_r = np.concatenate((uniq_r, uniq_r + num_rels))
    r_to_e = defaultdict(set)
    for s, r, _ in triplets:
        r_to_e[r].add(s)
        r_to_e[r + num_rels].add(s)
    r_len, e_idx, idx = [], [], 0
    for r in uniq_r:
        r_len.append((idx, idx + len(r_to_e[r])))
        e_idx.extend(r_to_e[r])
        idx += len(r_to_e[r])
    return uniq_r, r_len, e_idx

def _comp_deg_norm(g: dgl.DGLGraph):
    deg = g.in_degrees().float()
    deg[deg == 0] = 1
    return 1.0 / deg


def build_graph(num_nodes: int, num_rels: int, triples: np.ndarray,
                use_cuda: bool, gpu: int):
    _assert_ndarray(triples, 2, np.int64, 'triples')
    if triples.shape[1] != 4:
        raise ValueError('triples must have 4 cols (h,r,t,freq)')
    src, rel, dst, fre = triples.transpose()
    g = dgl.graph((src, dst), num_nodes=num_nodes)
    norm = _comp_deg_norm(g)
    g.ndata['id'] = torch.arange(num_nodes, dtype=torch.long).view(-1, 1)
    g.ndata['norm'] = norm.view(-1, 1)
    g.apply_edges(lambda e: {'norm': e.dst['norm'] * e.src['norm']})
    g.edata['type'] = torch.LongTensor(rel)
    g.edata['fre'] = torch.LongTensor(fre)
    return g.to(gpu) if use_cuda else g


def build_sub_graph(num_nodes: int, num_rels: int, triples: np.ndarray,
                    use_cuda: bool, gpu: int):
    _assert_ndarray(triples, 2, np.int64, 'triples')
    if triples.shape[1] != 3:
        raise ValueError('triples must have 3 cols (h,r,t)')
    src, rel, dst = triples.transpose()
    src2, dst2 = np.concatenate([src, dst]), np.concatenate([dst, src])
    rel2 = np.concatenate([rel, rel + num_rels])
    g = dgl.graph((src2, dst2), num_nodes=num_nodes)
    norm = _comp_deg_norm(g)
    g.ndata['id'] = torch.arange(num_nodes, dtype=torch.long).view(-1, 1)
    g.ndata['norm'] = norm.view(-1, 1)
    g.apply_edges(lambda e: {'norm': e.dst['norm'] * e.src['norm']})
    g.edata['type'] = torch.LongTensor(rel2)
    uniq_r, r_len, r_to_e = r2e(triples, num_rels)
    g.uniq_r, g.r_len, g.r_to_e = uniq_r, r_len, r_to_e
    return g.to(gpu) if use_cuda else g

def get_total_rank(test_triples: torch.Tensor,
                   score: torch.Tensor,
                   all_ans,
                   eval_bz: int,
                   rel_predict: int = 0,
                   train_set=None,
                   print_unseen: bool = False):
    _assert_tensor(score, 2, torch.float32, 'score')
    _assert_tensor(test_triples, 2, torch.long, 'test_triples')
    num = len(test_triples)
    ranks, filt_ranks = [], []
    for i in range(0, num, eval_bz):
        end = min(num, i + eval_bz)
        tb = test_triples[i:end]
        sb = score[i:end].clone()
        tgt = tb[:, 1] if rel_predict == 1 else tb[:, 0] if rel_predict == 2 else tb[:, 2]
        ranks.append(sort_and_rank(sb, tgt))
        filt = filter_score_r(tb, sb, all_ans) if rel_predict else filter_score(tb, sb, all_ans)
        filt_ranks.append(sort_and_rank(filt, tgt))

    rank = torch.cat(ranks) + 1
    filt_rank = torch.cat(filt_ranks) + 1
    mrr = torch.mean(1.0 / rank.float())
    filt_mrr = torch.mean(1.0 / filt_rank.float())

    if print_unseen and train_set is not None:
        unseen_mask = torch.tensor(
            [tuple(tr.cpu().tolist()) not in train_set for tr in test_triples],
            dtype=torch.bool, device=rank.device
        )
        unseen_rank = rank[unseen_mask]
        unseen_filt = filt_rank[unseen_mask]
        if unseen_rank.numel() == 0:
            unseen_mrr = unseen_filt_mrr = torch.tensor(0.0, device=rank.device)
        else:
            unseen_mrr = torch.mean(1.0 / unseen_rank.float())
            unseen_filt_mrr = torch.mean(1.0 / unseen_filt.float())
        return (filt_mrr.item(), mrr.item(), rank, filt_rank,
                unseen_rank, unseen_filt)
    return filt_mrr.item(), mrr.item(), rank, filt_rank

def flatten(l):
    out = []
    for c in l:
        out.extend(flatten(c) if isinstance(c, (list, tuple)) else [c])
    return out


def UnionFindSet(m: int, edges: List[Tuple[int, int]]):
    roots, rank_arr = list(range(m)), [0] * m

    def find(x):
        while roots[x] != x:
            roots[x] = roots[roots[x]]
            x = roots[x]
        return x

    for u, v in edges:
        pu, pv = find(u), find(v)
        if pu != pv:
            if rank_arr[pu] < rank_arr[pv]:
                roots[pu] = pv
            elif rank_arr[pu] > rank_arr[pv]:
                roots[pv] = pu
            else:
                roots[pv] = pu
                rank_arr[pu] += 1
            m -= 1
    return m


def append_object(e1, e2, r, d):
    d.setdefault(e1, {}).setdefault(r, set()).add(e2)


def add_subject(e1, e2, r, d, num_rel):
    d.setdefault(e2, {}).setdefault(r + num_rel, set()).add(e1)


def add_object(e1, e2, r, d, num_rel):
    d.setdefault(e1, {}).setdefault(r, set()).add(e2)


def load_all_answers(total_data: np.ndarray, num_rel: int):
    obj, subj = {}, {}
    for s, r, o in total_data[:, :3]:
        add_subject(s, o, r, subj, num_rel)
        add_object(s, o, r, obj, 0)
    return obj, subj


def load_all_answers_for_filter(total_data: np.ndarray, num_rel: int, rel_p: bool = False):
    def add_rel(e1, e2, r, d):
        d.setdefault(e1, {}).setdefault(e2, set()).add(r)

    ans = {}
    for s, r, o in total_data[:, :3]:
        if rel_p:
            add_rel(s, o, r, ans)
            add_rel(o, s, r + num_rel, ans)
        else:
            add_subject(s, o, r, ans, num_rel)
            add_object(s, o, r, ans, 0)
    return ans


def load_all_answers_for_time_filter(total_data, num_rels, num_nodes, rel_p=False):
    return [load_all_answers_for_filter(snap, num_rels, rel_p) for snap in split_by_time(total_data)]


def split_by_time(data: np.ndarray):
    snapshots, buf, last_t = [], [], None
    for triple in data:
        t = triple[3]
        if last_t is None or t == last_t:
            buf.append(triple[:3])
        else:
            snapshots.append(np.asarray(buf))
            buf = [triple[:3]]
        last_t = t
    if buf:
        snapshots.append(np.asarray(buf))
    return snapshots


def slide_list(snapshots: List[np.ndarray], k: int = 1):
    if k > len(snapshots):
        raise ValueError('history length exceeds snapshot list')
    for i in tqdm(range(len(snapshots) - k + 1), disable=True):
        yield snapshots[i:i + k]


def construct_snap(test_triples, num_nodes, num_rels, final_score, topK):
    _, idx = torch.sort(final_score, dim=1, descending=True)
    top_idx = idx[:, :topK]
    preds = []
    for row, id_list in enumerate(top_idx):
        h, r = test_triples[row, 0], test_triples[row, 1]
        for id_ in id_list:
            id_ = id_.item()
            if r < num_rels:
                preds.append([h.item(), r.item(), id_])
            else:
                preds.append([id_, r.item() - num_rels, h.item()])
    return np.asarray(preds, dtype=int)


def construct_snap_r(test_triples, num_nodes, num_rels, final_score, topK):
    _, idx = torch.sort(final_score, dim=1, descending=True)
    top_idx = idx[:, :topK]
    preds = []
    for row, id_list in enumerate(top_idx):
        h, t = test_triples[row, 0], test_triples[row, 2]
        for id_ in id_list:
            id_ = id_.item()
            if id_ < num_rels:
                preds.append([h.item(), id_, t.item()])
            else:
                preds.append([t.item(), id_ - num_rels, h.item()])
    return np.asarray(preds, dtype=int)


def dilate_input(input_list: List[np.ndarray], dilate_len: int):
    dilated, buf = [], []
    for i, arr in enumerate(input_list):
        if i and i % dilate_len == 0:
            dilated.append(np.unique(np.concatenate(buf), axis=0))
            buf = []
        buf.append(arr)
    dilated.append(np.unique(np.concatenate(buf), axis=0))
    return dilated


def emb_norm(emb: torch.Tensor, eps: float = 1e-5):
    _assert_tensor(emb, 2, torch.float32, 'emb')
    norm = torch.sqrt(torch.sum(emb.pow(2), dim=1)).clamp(min=eps)
    return emb / norm.unsqueeze(1)


def shuffle(data: np.ndarray, labels: np.ndarray):
    _assert_ndarray(data, 2, data.dtype, 'data')
    _assert_ndarray(labels, 1, labels.dtype, 'labels')
    idx = np.random.permutation(len(data))
    return data[idx], labels[idx]


def cuda(tensor: torch.Tensor):
    return tensor.cuda(non_blocking=True) if tensor.device.type == 'cpu' else tensor


def soft_max(z: np.ndarray):
    _assert_ndarray(z, 1, z.dtype, 'z')
    return _safe_softmax_numpy(z.astype(np.float64))
