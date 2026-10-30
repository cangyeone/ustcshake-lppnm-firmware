"""Cross-language bit-exactness, reset, boundary, and export checks."""
import json
from pathlib import Path
import subprocess
import numpy as np
import onnx
import onnxruntime as ort
import torch
from build import OUT, load_graph, full_integer, run_stream
from graph import load_model
from stream import CausalNormalizer, Picker


def check_native(name,mode,x,expected_bin,expected_cl,expected_tm,expected_emit):
    binary=OUT/(name+'.input.bin')
    csv=OUT/(name+'.output.csv')
    np.asarray(x,dtype='<f4').tofile(binary)
    p=subprocess.run([str(OUT/'stream_cli'),mode,str(binary),str(csv)],capture_output=True,text=True,check=True)
    actual=np.loadtxt(csv,delimiter=',',skiprows=1,dtype=np.int64,ndmin=2)
    expected=np.column_stack([expected_bin,expected_cl,expected_tm,expected_emit])
    if not np.array_equal(actual,expected):
        mismatch=np.argwhere(actual!=expected)
        raise AssertionError((name,actual.shape,expected.shape,mismatch[:10].tolist(),
                              actual[tuple(mismatch[0])],expected[tuple(mismatch[0])]))
    return json.loads(p.stderr)


def main():
    torch.set_num_threads(1)
    nodes,heads=load_graph()
    d=np.load(OUT/'validation_input.npz')
    report={}
    report['normalized']=check_native('normalized','--normalized',d['normalized'],d['bin'],d['class_int8'],d['time_int8'],d['emit_at'])
    valid=d['emit_at']>=3072
    report['raw_with_causal_normalizer']=check_native('raw','--raw',d['raw100'],d['bin'][valid],d['class_int8'][valid],d['time_int8'][valid],d['emit_at'][valid])
    # Exercise adversarial min/max queues (all equal, monotonic rise/fall), ring
    # wraparound, and step impulses in the native normalizer and graph runtime.
    waves=[np.zeros(4096),np.ones(4096),np.linspace(-10,10,4096),
           np.linspace(10,-10,4096),np.tile([-1,1],2048)]
    for i,wave in enumerate(waves):
        raw=wave.astype(np.float32)
        normalized=CausalNormalizer().process(raw)
        s,ids,outputs,emitted,_=run_stream(nodes,heads,normalized,True)
        keep=emitted>=3072
        report[f'edge_{i}']=check_native(f'edge_{i}','--raw',raw,ids[keep],outputs[0][keep],outputs[1][keep],emitted[keep])
    half=((np.arange(-127,127,dtype=np.float64)+0.5)/127).astype(np.float32)
    rounding=np.resize(np.concatenate([half,np.nextafter(half,np.float32(np.inf)),
                                       np.nextafter(half,np.float32(-np.inf))]),(4096,3))
    _,ri,ro,re,_=run_stream(nodes,heads,rounding,True)
    report['quantization_half_steps']=check_native('rounding','--normalized',rounding,ri,ro[0],ro[1],re)
    m=onnx.load(OUT/'lppnm.fused.3072.onnx');onnx.checker.check_model(m)
    assert not any(n.op_type=='BatchNormalization' for n in m.graph.node)
    x=d['normalized'][16384:16384+3072].T[None].copy()
    y=ort.InferenceSession(str(OUT/'lppnm.fused.3072.onnx'),providers=['CPUExecutionProvider']).run(None,{'normalized_wave':x})
    _,fused=load_model()
    with torch.inference_mode():z=fused(torch.from_numpy(x))
    assert all(np.allclose(a,b.numpy(),atol=1e-4,rtol=1e-4) for a,b in zip(y,z))
    report['onnx_valid_and_matches_fused']=True
    report['all_native_int8_outputs_bit_exact']=True
    report['native_normalization_matches_python']=True
    report['emission_delay_seconds_range']=[float((d['emit_at']-d['bin']*8).min()/100),float((d['emit_at']-d['bin']*8).max()/100)]
    (OUT/'native_validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
