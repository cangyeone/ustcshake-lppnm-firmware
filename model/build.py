"""Fuse, calibrate W8A8, export portable weights, and test incremental execution.

Run from workspace: .pnsn-venv/bin/python experiments/lppnm_streaming_3072/build.py
"""
import hashlib
import os
import json
from pathlib import Path
import time

import numpy as np
import obspy
import torch
from torch.nn import functional as F

from graph import Node, REPO, load_model, make_graph, full_float, quantize, dependency, round_away
from stream import Stream, Picker, CausalNormalizer, requant, quantize_input, decode, WINDOW, HOP

HERE = Path(__file__).resolve().parent
OUT = HERE / 'artifacts'


def save_graph(nodes,heads):
    arrays = {}
    meta = []
    for i,n in enumerate(nodes):
        item = dict(vars(n))
        for key in ('weight','bias','qweight','qbias','multiplier'):
            value = item.pop(key)
            if value is not None:
                arrays[f'n{i}_{key}'] = value
        meta.append(item)
    np.savez(OUT/'weights.npz',**arrays)
    (OUT/'graph.json').write_text(json.dumps({'nodes':meta,'heads':heads},indent=2))


def load_graph(directory=OUT):
    meta = json.loads((directory/'graph.json').read_text())
    arrays = np.load(directory/'weights.npz',allow_pickle=False)
    nodes = []
    for i,item in enumerate(meta['nodes']):
        item['parents'] = tuple(item['parents'])
        for key in ('weight','bias','qweight','qbias','multiplier'):
            k = f'n{i}_{key}'
            if k in arrays:
                item[key] = arrays[k].copy()
        nodes.append(Node(**item))
    return nodes,tuple(meta['heads'])


def full_integer(nodes,x):
    """Independent offline reference; float64 convolution is exact for these
    integer accumulations (<2**31), then explicit Q24 integer requantization.
    This reference is for testing, not for deployment.
    """
    values = [quantize_input(x)]
    for n in nodes[1:]:
        a = values[n.parents[0]]
        if n.kind == 'conv':
            acc = F.conv1d(torch.from_numpy(a.T[None].astype(np.float64)),
                          torch.from_numpy(n.qweight.astype(np.float64)),
                          torch.from_numpy(n.qbias.astype(np.float64)),
                          stride=n.stride,padding=n.padding,groups=n.groups)[0].T.numpy()
            assert np.max(np.abs(acc)) <= np.iinfo(np.int32).max
            y = requant(acc.astype(np.int64)*n.multiplier)
            if n.relu:
                y = np.maximum(y,0)
        elif n.kind == 'up':
            y = np.repeat(a,2,axis=0)
        elif n.kind == 'add':
            y = requant(a.astype(np.int64)*n.multiplier[0]+values[n.parents[1]].astype(np.int64)*n.multiplier[1])
        elif n.kind == 'cat':
            y = np.concatenate([requant(values[p].astype(np.int64)*mul) for p,mul in zip(n.parents,n.multiplier)],axis=1)
        values.append(y)
    return values


def run_stream(nodes,heads,x,integer):
    s = Stream(nodes,heads,integer)
    ids,cl,tm,emit_at,work = [],[],[],[],[]
    for start in range(0,len(x),HOP):
        old = s.macs
        idx,values = s.process(x[start:start+HOP])
        ids.extend(idx);cl.extend(values[0]);tm.extend(values[1])
        emit_at.extend([start+HOP]*len(idx))
        work.append(s.macs-old)
    return s,np.array(ids,dtype=np.int64),[np.array(cl),np.array(tm)],np.array(emit_at),work


def main():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    OUT.mkdir(exist_ok=True)
    source = Path(os.environ['LPPNM_CALIBRATION_WAVEFORM']) if 'LPPNM_CALIBRATION_WAVEFORM' in os.environ else next((REPO/'data/waveform').glob('*53085*BHZ*.sac'))
    trace = obspy.read(str(source))[0]
    assert trace.stats.sampling_rate == 100
    raw = trace.data[:25600].astype(np.float32)
    x = CausalNormalizer().process(raw)
    original,fused = load_model()
    nodes,heads = make_graph(fused)
    assert len([n for n in nodes if n.kind=='conv']) == 39
    assert not any(isinstance(m,torch.nn.BatchNorm2d) for m in fused.modules())
    sample = torch.from_numpy(x[:WINDOW].T[None].copy())
    with torch.inference_mode():
        a,b = original(sample),fused(sample)
    assert torch.allclose(a[0],b[0],atol=1e-5,rtol=1e-4)
    assert torch.allclose(a[1],b[1],atol=1e-4,rtol=1e-4)
    ref = full_float(nodes,x)
    reference = decode(nodes,heads,np.arange(len(ref[heads[0]])),[ref[p] for p in heads])
    with torch.inference_mode():
        whole = fused(torch.from_numpy(x.T[None].copy()))
    assert np.allclose(reference['prob'].T[None],whole[0].numpy(),atol=1e-5,rtol=1e-4)
    fs,ids,fo,emitted_at,_ = run_stream(nodes,heads,x,False)
    assert np.array_equal(ids,np.arange(len(ids)))
    float_error = max(float(np.abs(v-ref[p][ids]).max()) for v,p in zip(fo,heads))
    assert all(np.allclose(v,ref[p][ids],atol=2e-4,rtol=2e-4) for v,p in zip(fo,heads)),float_error
    lookahead = [dependency(nodes,heads[0],j)[1]-8*j for j in range(64,128)]
    # Check every emitted output had all true future context, not padded guesses.
    assert all(dependency(nodes,heads[0],int(j))[1] < t for j,t in zip(ids,emitted_at))

    quantize(nodes,[x[s:s+WINDOW] for s in (0,3072,6144,9216)])
    qref = full_integer(nodes,x)
    qs,qids,qo,qemitted,work = run_stream(nodes,heads,x,True)
    assert np.array_equal(qids,ids)
    assert all(np.array_equal(v,qref[p][ids]) for v,p in zip(qo,heads)), 'INT8 streaming must be bit exact'
    save_graph(nodes,heads)
    loaded,lh = load_graph()
    ls,li,lo,_,_ = run_stream(loaded,lh,x[:4096],True)
    assert all(np.array_equal(v,qref[p][li]) for v,p in zip(lo,heads))

    # Test reset and deterministic replay, constant signal and input rejection.
    picker = Picker(nodes,heads)
    first = [picker.push(raw[s:s+HOP]) for s in range(0,4096,HOP)]
    picker.reset()
    second = [picker.push(raw[s:s+HOP]) for s in range(0,4096,HOP)]
    assert all(np.array_equal(a['prob'],b['prob']) for a,b in zip(first,second))
    assert all(len(r['bin'])==0 for r in first[:23])
    before = picker.normalizer.count
    for invalid in (np.zeros(127),np.full(HOP,np.nan),np.full(HOP,np.inf)):
        try:
            picker.push(invalid)
            raise AssertionError('invalid input accepted')
        except ValueError:
            pass
    assert picker.normalizer.count == before
    cn = CausalNormalizer()
    assert np.array_equal(cn.process(np.ones(4096)),np.zeros((4096,3)))

    # Synthetic data independently exercises signs, impulses and all phases of
    # the stride-64 encoder / nearest-neighbour decoder alignment.
    rng = np.random.default_rng(42)
    test = rng.uniform(-1,1,(8192,3)).astype(np.float32)
    test[4096] = (1,-1,1)
    sr = full_integer(nodes,test)
    ss,si,so,_,_ = run_stream(nodes,heads,test,True)
    assert all(np.array_equal(v,sr[p][si]) for v,p in zip(so,heads))

    fd = decode(nodes,heads,ids,fo)
    qd = decode(nodes,heads,ids,qo,True)
    held = ids*8 >= 14336
    delta = np.abs(fd['prob'][held]-qd['prob'][held])
    # Quantify changed preprocessing separately from quantization; not F1.
    raw_window = raw[16384:16384+WINDOW]
    legacy = (raw_window-raw_window.mean())/(np.max(np.abs(raw_window-raw_window.mean()))+1e-6)
    with torch.inference_mode():
        lp = fused(torch.from_numpy(np.tile(legacy,(1,3,1))))[0].numpy()
        cp = fused(torch.from_numpy(x[16384:16384+WINDOW].T[None].copy()))[0].numpy()
    torch.save(fused.state_dict(),OUT/'lppnm.fused.pt')
    torch.onnx.export(fused,(sample,),str(OUT/'lppnm.fused.3072.onnx'),dynamo=False,
                      opset_version=13,input_names=['normalized_wave'],output_names=['prob','offset'])
    frames = [max(a,b)+2 for a,b in zip(qs.max_frames,ss.max_frames)]
    (OUT/'capacities.json').write_text(json.dumps(frames))
    report = {
        'window':WINDOW,'hop':HOP,'sample_rate_hz':100,'classes':7,
        'normalization':'causal trailing-window mean/centered-maxabs, old samples immutable',
        'weight_quantization':'INT8 symmetric per-output-channel',
        'activation_quantization':'INT8 symmetric per-tensor; INT32 accumulator; INT64 Q24 requantization',
        'fp32_postprocessing':['softmax over all 7 classes','sigmoid time offset'],
        'fused_bn_remaining':0,'quantized_convolutions':39,
        'checkpoint_sha256':hashlib.sha256((REPO/'ckpt/china.lppnm.pt').read_bytes()).hexdigest(),
        'calibration_source':str(source),'calibration_samples':[0,12288],
        'evaluation_samples':[14336,25600],
        'float_stream_vs_full_max_abs_logit_or_offset_logit':float_error,
        'int8_stream_vs_full_bit_exact':True,'synthetic_alignment_bit_exact':True,
        'reset_replay_exact':True,'invalid_input_no_state_mutation':True,
        'lookahead_samples_min':min(lookahead),'lookahead_samples_max':max(lookahead),
        'output_grid_samples':8,'steady_conv_macs_per_128':work[-1],
        'steady_conv_macs_per_second':work[-1]*100/HOP,
        'full_3072_conv_macs':15148032,
        'stream_cache_peak_bytes_python_int8':qs.peak_cached_bytes,
        'portable_cache_allocated_bytes':sum(c*n.channels for c,n in zip(frames,nodes)),
        'raw_history_float32_bytes':WINDOW*4,
        'int8_weight_bytes':sum(n.qweight.nbytes for n in nodes if n.qweight is not None),
        'int32_bias_bytes':sum(n.qbias.nbytes for n in nodes if n.qbias is not None),
        'int64_multiplier_bytes':sum(n.multiplier.nbytes for n in nodes if n.multiplier is not None),
        'heldout_quantization':{'prob_mae':float(delta.mean()),'prob_max_abs':float(delta.max()),
            'offset_max_error_ms':float(np.abs(fd['sample'][held]-qd['sample'][held]).max()*10),
            'p_s_threshold_0_3_bin_disagreements':int(((fd['prob'][held,1:3]>.3)!=(qd['prob'][held,1:3]>.3)).sum()),
            'p_s_bins':int(held.sum()*2)},
        'causal_vs_legacy_normalization_probability_mae_one_window':float(np.abs(lp-cp).mean()),
        'limitations':['single record; no labeled precision/recall validation',
                      'no ESP32-C3 timing or total SRAM measurement',
                      'lookahead and warmup are intentional; no fabricated right-boundary predictions',
                      'portable custom quantization, not an ESP-DL .espdl artifact']}
    (OUT/'validation.json').write_text(json.dumps(report,indent=2))
    np.savez(OUT/'validation_input.npz',raw100=raw,normalized=x,bin=ids,
             class_int8=qo[0],time_int8=qo[1],emit_at=qemitted)
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
