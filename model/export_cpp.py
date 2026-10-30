"""Export calibrated model as readonly C++ arrays; no host ML runtime on device."""
import json
from pathlib import Path
from build import load_graph, OUT


def main():
    nodes,heads=load_graph()
    capacities=json.loads((OUT/'capacities.json').read_text())
    lines=['#pragma once', '#include "../runtime.hpp"', 'namespace lppnm_model {']
    for i,n in enumerate(nodes):
        for key,typ in [('qweight','int8_t'),('qbias','int32_t'),('multiplier','int64_t')]:
            a=getattr(n,key)
            if a is None: continue
            lines.append(f'alignas(16) inline const {typ} n{i}_{key}[] = {{')
            flat=a.flatten()
            for j in range(0,len(flat),24):
                lines.append(','.join(str(int(v)) for v in flat[j:j+24])+',')
            lines.append('};')
    kinds={'input':'Input','conv':'Conv','up':'Up','add':'Add','cat':'Cat'}
    lines.append('inline const lppnm::NodeSpec nodes[] = {')
    for i,n in enumerate(nodes):
        parents=list(n.parents)+[-1,-1]
        k=n.weight.shape[2] if n.kind=='conv' else 1
        ptr=lambda key:f'n{i}_{key}' if getattr(n,key) is not None else 'nullptr'
        scale=f'{n.scale:.10e}f'
        fields=[f'lppnm::{kinds[n.kind]}',str(parents[0]),str(parents[1]),str(n.channels),str(k),
                str(n.stride),str(n.padding),str(n.groups),'true' if n.relu else 'false',
                str(capacities[i]),scale,ptr('qweight'),ptr('qbias'),ptr('multiplier')]
        lines.append('{'+','.join(fields)+'}, // '+n.name)
    lines.extend(['};',f'constexpr int count={len(nodes)}, class_head={heads[0]}, time_head={heads[1]};','}'])
    (OUT/'model.hpp').write_text('\n'.join(lines)+'\n')
    print('Exported',len(nodes),'nodes to',OUT/'model.hpp')


if __name__=='__main__':main()
