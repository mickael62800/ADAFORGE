"""Experimental FFN Q8 comparison; explicitly reports changed token IDs."""
import argparse,json,os,statistics,subprocess
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('model',type=Path);args=p.parse_args()
root=Path(__file__).resolve().parents[1]
results={}
for label,enabled in [('fp32_before',False),('q8_experimental',True),('fp32_after',False)]:
 env=os.environ.copy();env.pop('ADAFORGE_EXPERIMENTAL_Q8',None)
 if enabled:env['ADAFORGE_EXPERIMENTAL_Q8']='1'
 command=[str(root/'target/release/adaforge.exe'),'bench',str(args.model.resolve()),'Bonjour, comment vas-tu ?','24','5']
 result=subprocess.run(command,env=env,capture_output=True,encoding='utf-8',check=True)
 rows=[json.loads(line.removeprefix('BENCH_SAMPLE '))for line in result.stdout.splitlines()if line.startswith('BENCH_SAMPLE ')]
 assert len(rows)==5 and all(len(x['generated_ids'])==24 for x in rows)
 assert all(x['generated_ids']==rows[0]['generated_ids'] for x in rows)
 results[label]={'median_tps':statistics.median(x['decode_tokens']*1000/x['decode_ms']for x in rows),'samples':rows}
 print(label,results[label]['median_tps'],flush=True)
assert all(x['generated_ids']==results['fp32_before']['samples'][0]['generated_ids']for x in results['fp32_after']['samples'])
results['identical_tokens_between_precision_modes']=results['fp32_before']['samples'][0]['generated_ids']==results['q8_experimental']['samples'][0]['generated_ids']
results['protocol']={'down_kernel':'q4_q8_matvec','scope':'FFN only; Q4 weights, INT8 activations with FP32 per-32 scale; other operations FP32','output_tokens':24,'runs':5,'warmups':1,'kv_reset':True,'quality_validation':'kernel CPU oracle verified, full-model quality NOT established','selection':'greedy; token IDs may differ between precision modes'}
path=root/'benchmarks/q8-experimental.json';path.write_text(json.dumps(results,indent=2),encoding='utf-8');print('Report:',path,flush=True)
