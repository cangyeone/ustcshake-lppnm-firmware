"""True incremental FP32/INT8 reference runtime; no whole-window CNN replay."""
from collections import deque
import numpy as np
from graph import round_away

WINDOW = 3072
HOP = 128


class CausalNormalizer:
    """Trailing 3072 samples, mean and centered maxabs, emitted once per sample.

    Unlike rolling-window normalization, normalized past samples never change.
    O(1) amortized min/max queues; single waveform history, not three copies.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.history = deque()
        self.low = deque()
        self.high = deque()
        self.total = 0.0
        self.count = 0

    def process(self, samples):
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim != 1 or not np.isfinite(samples).all():
            raise ValueError('Expected finite one-dimensional 100 Hz waveform')
        out = np.empty(len(samples), dtype=np.float32)
        for k, v in enumerate(samples):
            v = float(v)
            i = self.count
            self.count += 1
            self.history.append(v)
            self.total += v
            if len(self.history) > WINDOW:
                self.total -= self.history.popleft()
            while self.low and self.low[-1][1] >= v:
                self.low.pop()
            while self.high and self.high[-1][1] <= v:
                self.high.pop()
            self.low.append((i,v)); self.high.append((i,v))
            while self.low[0][0] <= i-WINDOW:
                self.low.popleft()
            while self.high[0][0] <= i-WINDOW:
                self.high.popleft()
            mean = self.total / len(self.history)
            scale = max(abs(self.low[0][1]-mean), abs(self.high[0][1]-mean)) + 1e-6
            out[k] = (v-mean)/scale
        return np.repeat(out[:,None], 3, axis=1)


def requant(value):
    """Round a Q24 signed integer half away from zero, then saturate INT8."""
    value = np.asarray(value, dtype=np.int64)
    v = np.where(value >= 0, (value+(1<<23)) >> 24, -((-value+(1<<23)) >> 24))
    return v.clip(-127,127).astype(np.int8)


def quantize_input(x):
    # Explicit FP64 product matches the portable C++ rounding at half steps.
    return round_away(np.asarray(x,dtype=np.float64)*127.0).clip(-127,127).astype(np.int8)


class Stream:
    def __init__(self, nodes, heads, integer=False):
        self.nodes = nodes
        self.heads = heads
        self.integer = integer
        self.reset()

    def reset(self):
        dtype = np.int8 if self.integer else np.float32
        self.buffers = [np.empty((0,n.channels),dtype=dtype) for n in self.nodes]
        self.base = [0]*len(self.nodes)
        self.end = [0]*len(self.nodes)
        self.next = [0]*len(self.nodes)
        self.emitted = 0
        self.max_frames = [0]*len(self.nodes)
        self.macs = 0
        self.peak_cached_bytes = 0

    def ready(self, i):
        n = self.nodes[i]
        if n.kind == 'conv':
            # All right context must have arrived. No artificial right padding.
            return max(0, (self.end[n.parents[0]]-1-n.padding)//n.stride+1)
        if n.kind == 'up':
            return self.end[n.parents[0]]*2
        return min(self.end[p] for p in n.parents)

    def get(self, p, ids):
        ids = np.asarray(ids, dtype=np.int64)
        x = self.buffers[p]
        result = np.zeros((len(ids),self.nodes[p].channels),dtype=x.dtype)
        valid = ids >= 0
        if valid.any():
            local = ids[valid]-self.base[p]
            if local.min() < 0 or local.max() >= len(x):
                raise RuntimeError(('cache alignment',p,ids,self.base[p],self.end[p]))
            result[valid] = x[local]
        return result

    def append(self, i, data):
        if len(data):
            self.buffers[i] = np.concatenate((self.buffers[i],data),axis=0)
            self.end[i] += len(data)
            self.max_frames[i] = max(self.max_frames[i],len(self.buffers[i]))
            self.peak_cached_bytes = max(self.peak_cached_bytes,sum(b.nbytes for b in self.buffers))

    def process(self, normalized):
        x = np.asarray(normalized,dtype=np.float32)
        if x.shape != (HOP,3) or not np.isfinite(x).all():
            raise ValueError('Each call requires exactly [128,3] finite normalized 100 Hz samples')
        if self.integer:
            x = quantize_input(x)
        self.append(0,x)
        for i,n in enumerate(self.nodes[1:],1):
            stop = self.ready(i)
            ids = np.arange(self.next[i],stop,dtype=np.int64)
            if not len(ids):
                continue
            p = n.parents[0]
            if n.kind == 'conv':
                k = n.weight.shape[2]
                positions = ids[:,None]*n.stride-n.padding+np.arange(k)
                patch = self.get(p,positions.flatten()).reshape(len(ids),k,-1).transpose(0,2,1)
                in_per = n.weight.shape[1]
                out_per = n.channels//n.groups
                grouped = patch.reshape(len(ids),n.groups,in_per*k)
                if self.integer:
                    w = n.qweight.reshape(n.groups,out_per,in_per*k).astype(np.int64)
                    acc = np.einsum('ngi,goi->ngo',grouped.astype(np.int64),w).reshape(len(ids),n.channels)
                    acc += n.qbias
                    if np.abs(acc).max() > np.iinfo(np.int32).max:
                        raise OverflowError(n.name)
                    y = requant(acc*n.multiplier)
                else:
                    w = n.weight.reshape(n.groups,out_per,in_per*k)
                    y = np.einsum('ngi,goi->ngo',grouped,w).reshape(len(ids),n.channels)+n.bias
                if n.relu:
                    y = np.maximum(y,0)
                self.macs += len(ids)*n.weight.size
            elif n.kind == 'up':
                y = self.get(p,ids//2)
            elif n.kind == 'add':
                a,b = [self.get(p,ids) for p in n.parents]
                y = requant(a.astype(np.int64)*n.multiplier[0]+b.astype(np.int64)*n.multiplier[1]) if self.integer else a+b
            elif n.kind == 'cat':
                parts = [self.get(p,ids) for p in n.parents]
                if self.integer:
                    parts = [requant(v.astype(np.int64)*mul) for v,mul in zip(parts,n.multiplier)]
                y = np.concatenate(parts,axis=1)
            else:
                raise ValueError(n.kind)
            self.append(i,y)
            self.next[i] = stop
        stop = min(self.end[p] for p in self.heads)
        ids = np.arange(self.emitted,stop,dtype=np.int64)
        outputs = [self.get(p,ids) for p in self.heads]
        self.emitted = stop
        self.trim()
        return ids, outputs

    def trim(self):
        needed = list(self.end)
        for i,n in enumerate(self.nodes[1:],1):
            if n.kind == 'conv':
                start = max(0,self.next[i]*n.stride-n.padding)
            elif n.kind == 'up':
                start = self.next[i]//2
            else:
                start = self.next[i]
            for p in n.parents:
                needed[p] = min(needed[p],start)
        for p in self.heads:
            needed[p] = min(needed[p],self.emitted)
        for i,start in enumerate(needed):
            drop = max(0,start-self.base[i])
            self.buffers[i] = self.buffers[i][drop:].copy()
            self.base[i] += drop


def decode(nodes, heads, ids, outputs, integer=False):
    cl, tm = [v.astype(np.float32) for v in outputs]
    if integer:
        cl *= nodes[heads[0]].scale
        tm *= nodes[heads[1]].scale
    ex = np.exp(cl-cl.max(axis=1,keepdims=True))
    prob = ex/ex.sum(axis=1,keepdims=True)  # Keep all seven classes.
    offsets = 8/(1+np.exp(-np.clip(tm[:,0],-80,80)))
    return {'bin': ids, 'sample': ids*8+offsets, 'prob': prob}


class Picker:
    """100 Hz input only. reset() is mandatory on gaps or a new device session."""
    def __init__(self,nodes,heads,integer=True):
        self.normalizer = CausalNormalizer()
        self.network = Stream(nodes,heads,integer)

    def reset(self):
        self.normalizer.reset()
        self.network.reset()

    def push(self,raw128):
        raw128 = np.asarray(raw128,dtype=np.float32)
        if raw128.shape != (HOP,) or not np.isfinite(raw128).all():
            raise ValueError('Expected exactly 128 finite samples at 100 Hz')
        ids,outputs = self.network.process(self.normalizer.process(raw128))
        if self.normalizer.count < WINDOW:
            ids = ids[:0]
            outputs = [v[:0] for v in outputs]
        return decode(self.network.nodes,self.network.heads,ids,outputs,self.network.integer)
