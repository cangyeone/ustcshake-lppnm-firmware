"""Flatten the fused pretrained LPPNM into a time-indexed convolution DAG."""
from dataclasses import dataclass
from pathlib import Path
import copy
import sys
import os

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get('PNSN_REPO', ROOT / 'third_party/pnsn'))


@dataclass
class Node:
    name: str
    kind: str
    parents: tuple
    channels: int
    weight: object = None
    bias: object = None
    stride: int = 1
    padding: int = 0
    groups: int = 1
    relu: bool = False
    scale: float = 1.0
    qweight: object = None
    qbias: object = None
    multiplier: object = None


def load_model():
    sys.path.insert(0, str(REPO))
    from models.LPPNM import Model
    m = Model().eval()
    m.load_state_dict(torch.load(REPO / 'ckpt/china.lppnm.pt',
                                map_location='cpu', weights_only=True), strict=True)
    fused = copy.deepcopy(m)
    fused.fuse_model()
    return m, fused


def make_graph(model):
    nodes = [Node('input', 'input', (), 3)]

    def add(node):
        nodes.append(node)
        return len(nodes)-1

    def visit(module, parent, name):
        if module.__class__.__name__ == 'QInvertedResidual':
            out = visit(module.conv, parent, name+'.conv')
            if module.use_res:
                out = add(Node(name+'.add', 'add', (parent, out), nodes[out].channels))
            return out
        if isinstance(module, torch.ao.nn.intrinsic.ConvReLU2d):
            out = visit(module[0], parent, name+'.conv')
            nodes[out].relu = True
            return out
        if isinstance(module, nn.Conv2d):
            assert module.kernel_size[0] == 1 and module.padding[0] == 0
            return add(Node(name, 'conv', (parent,), module.out_channels,
                            module.weight.detach().numpy()[:,:,0,:].copy(),
                            module.bias.detach().numpy().copy(),
                            module.stride[1], module.padding[1], module.groups))
        if isinstance(module, nn.UpsamplingNearest2d):
            assert tuple(module.scale_factor) == (1.0, 2.0)
            return add(Node(name, 'up', (parent,), nodes[parent].channels))
        if isinstance(module, nn.Identity):
            return parent
        if isinstance(module, nn.Sequential):
            for key, child in module.named_children():
                parent = visit(child, parent, name+'.'+key)
            return parent
        raise TypeError((name, type(module)))

    a = visit(model.layers, 0, 'layers')
    b = visit(model.class_encoder, a, 'class_encoder')
    both = add(Node('features', 'cat', (a,b), nodes[a].channels+nodes[b].channels))
    cl = visit(model.cl, both, 'cl')
    tm = visit(model.tm, both, 'tm')
    return nodes, (cl, tm)


@torch.inference_mode()
def full_float(nodes, x):
    """Independent full-sequence evaluator; x has shape [N, 3]."""
    values = [torch.as_tensor(x, dtype=torch.float32).T[None]]
    for n in nodes[1:]:
        a = values[n.parents[0]]
        if n.kind == 'conv':
            y = F.conv1d(a, torch.from_numpy(n.weight), torch.from_numpy(n.bias),
                         stride=n.stride, padding=n.padding, groups=n.groups)
            if n.relu:
                y = y.relu()
        elif n.kind == 'up':
            y = a.repeat_interleave(2, dim=2)
        elif n.kind == 'add':
            y = a + values[n.parents[1]]
        elif n.kind == 'cat':
            y = torch.cat([values[p] for p in n.parents], dim=1)
        else:
            raise ValueError(n.kind)
        values.append(y)
    return [v[0].T.numpy().copy() for v in values]


def round_away(x):
    return np.sign(x) * np.floor(np.abs(x)+0.5)


def quantize(nodes, calibration):
    """Static symmetric W8A8; per-output-channel weights, INT32 bias/accumulator.

    Q24 integer multipliers implement requantization in the portable runtime.
    Output sigmoid and seven-class softmax remain FP32 outside the CNN.
    """
    maxima = np.zeros(len(nodes))
    for x in calibration:
        values = full_float(nodes, x)
        for i, v in enumerate(values):
            maxima[i] = max(maxima[i], float(np.abs(v).max()))
    for i, n in enumerate(nodes):
        n.scale = max(maxima[i], 1e-6)/127
        if n.kind == 'input':
            n.scale = 1/127  # causal normalization is bounded to [-1,1]
        if n.kind == 'up':
            n.scale = nodes[n.parents[0]].scale
        if n.kind == 'conv':
            ws = np.maximum(np.abs(n.weight).max(axis=(1,2)), 1e-10)/127
            n.qweight = round_away(n.weight/ws[:,None,None]).clip(-127,127).astype(np.int8)
            ps = nodes[n.parents[0]].scale
            qb = round_away(n.bias/(ps*ws))
            bound = np.abs(qb) + np.abs(n.qweight.astype(np.int64)).sum(axis=(1,2))*127
            if np.any(bound > np.iinfo(np.int32).max):
                raise OverflowError('INT32 accumulator bound: '+n.name)
            n.qbias = qb.astype(np.int32)
            n.multiplier = round_away(ps*ws/n.scale*(1<<24)).astype(np.int64)
        elif n.kind in ('add','cat'):
            n.multiplier = round_away(np.array([nodes[p].scale/n.scale for p in n.parents])*(1<<24)).astype(np.int64)
        if n.multiplier is not None and np.max(np.abs(n.multiplier)) * (1<<31) >= (1<<63):
            raise OverflowError('INT64 requantization bound: '+n.name)
    return maxima


def dependency(nodes, node_id, index, memo=None):
    """Exact input dependency interval for an interior output, before padding."""
    if memo is None:
        memo = {}
    key = (node_id,index)
    if key in memo:
        return memo[key]
    n = nodes[node_id]
    if n.kind == 'input':
        ans = (index,index)
    else:
        if n.kind == 'conv':
            ids = [index*n.stride-n.padding, index*n.stride-n.padding+n.weight.shape[2]-1]
        elif n.kind == 'up':
            ids = [index//2]
        else:
            ids = [index]
        spans = [dependency(nodes,p,j,memo) for p in n.parents for j in ids]
        ans = (min(s[0] for s in spans), max(s[1] for s in spans))
    memo[key] = ans
    return ans
