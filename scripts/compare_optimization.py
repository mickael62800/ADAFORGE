"""Paired unprofiled before/after measurements; require identical token IDs."""
import argparse,json,subprocess,statistics,os
from pathlib import Path
root=Path(__file__).resolve().parents[1]
model=root/'models/gemma-4-12b-it-qat-q4_0/gemma-4-12B-it-QAT-Q4_0.gguf'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--baseline',type=Path,default=root/'tools/optimization-baseline/adaforge.exe')
parser.add_argument('--output',type=Path,default=root/'benchmarks/optimization-before-after.json')
parser.add_argument('--prompt',default='Bonjour, comment vas-tu ?')
parser.add_argument('--tokens',type=int,default=24)
parser.add_argument('--runs',type=int,default=5)
parser.add_argument('--discard-leading',type=int,default=0,help='Extra measured requests discarded before each retained series')
parser.add_argument('--abba',action='store_true',help='Use baseline/candidate/candidate/baseline to balance linear drift')
parser.add_argument('--gpu-telemetry',action='store_true',help='Record nvidia-smi before and after each process, outside token timings')
args=parser.parse_args()
if not 2<=args.tokens<=128 or not 3<=args.runs<=20:parser.error('tokens: 2..128; runs: 3..20')
if args.discard_leading<0 or args.runs+args.discard_leading>20:parser.error('discard-leading >= 0; runs + discard-leading <= 20')
if os.environ.get('ADAFORGE_EXPERIMENTAL_Q8')=='1':parser.error('Unset ADAFORGE_EXPERIMENTAL_Q8; this comparison requires the FP32 path.')
def gpu_snapshot():
 if not args.gpu_telemetry:return None
 try:
  raw=subprocess.check_output(['nvidia-smi','--query-gpu=name,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,memory.used','--format=csv,noheader'],encoding='utf-8',timeout=10)
  return {'query':'name,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,memory.used','raw_csv':raw.strip(),'scope':'Boundary snapshot outside timed requests; not a continuous trace.'}
 except (OSError,subprocess.SubprocessError) as error:return {'error':str(error)}
results={}
candidate=root/'target/release/adaforge.exe'
sequence=[('before',args.baseline),('after',candidate)]
if args.abba:sequence.append(('after_repeat',candidate))
sequence.append(('before_repeat',args.baseline))
for name,exe in sequence:
 gpu_before=gpu_snapshot()
 r=subprocess.run([str(exe),'bench',str(model),args.prompt,str(args.tokens),str(args.runs+args.discard_leading)],capture_output=True,encoding='utf-8',check=True)
 gpu_after=gpu_snapshot()
 rows=[json.loads(x.removeprefix('BENCH_SAMPLE ')) for x in r.stdout.splitlines() if x.startswith('BENCH_SAMPLE ')]
 assert len(rows)==args.runs+args.discard_leading
 discarded=rows[:args.discard_leading];rows=rows[args.discard_leading:]
 assert all(x['decode_tokens']>0 and x['decode_ms']>0 for x in rows)
 results[name]={'samples':rows,'discarded_samples':discarded,'median_tps':statistics.median(x['decode_tokens']*1000/x['decode_ms'] for x in rows),'median_ttft_ms':statistics.median(x['ttft_ms'] for x in rows)}
 if args.gpu_telemetry:results[name]['gpu_telemetry']={'before':gpu_before,'after':gpu_after}
 print(name,results[name]['median_tps'],flush=True)
reference=results['before']['samples'][0]
assert all(x['generated_ids']==reference['generated_ids'] and x['prompt_ids']==reference['prompt_ids'] for group in results.values() for x in group['samples'])
control_rates=[results[label]['median_tps'] for label in ['before','before_repeat']]
candidate_rates=[results[label]['median_tps'] for label in ['after','after_repeat'] if label in results]
results['comparison']={'control_series_medians':control_rates,'candidate_series_medians':candidate_rates,'relative_delta_percent':100*(statistics.mean(candidate_rates)/statistics.mean(control_rates)-1),'control_drift_percent':100*(control_rates[-1]/control_rates[0]-1),'interpretation':'Descriptive paired results only; ABBA balances linear drift but does not eliminate nonlinear drift or establish statistical significance.'}
results['protocol']={'prompt':args.prompt,'output_tokens':args.tokens,'runs_per_process':args.runs,'warmups':1+args.discard_leading,'discard_leading':args.discard_leading,'sequence':[label for label,_ in sequence],'kv_reset':True,'identical_token_ids':True}
results['protocol']['baseline_binary']=str(args.baseline.resolve())
results['protocol']['candidate_binary']=str((root/'target/release/adaforge.exe').resolve())
results['protocol']['tiled_gate_environment_override']=os.environ.get('ADAFORGE_EXPERIMENTAL_TILED_GATE')
results['protocol']['kpar_projections_environment_override']=os.environ.get('ADAFORGE_EXPERIMENTAL_KPAR_PROJECTIONS')
results['protocol']['fused_gelu_environment_override']=os.environ.get('ADAFORGE_EXPERIMENTAL_FUSED_GELU')
results['protocol']['stream_weights_environment_override']=os.environ.get('ADAFORGE_EXPERIMENTAL_STREAM_WEIGHTS')
results['protocol']['q6_rows_environment_override']=os.environ.get('ADAFORGE_EXPERIMENTAL_Q6_ROWS')
results['protocol']['q6_tiles_environment_override']=os.environ.get('ADAFORGE_EXPERIMENTAL_Q6_TILES')
results['protocol']['q6_raw_environment_override']=os.environ.get('ADAFORGE_EXPERIMENTAL_Q6_RAW')
results['protocol']['norm_rope_environment_override']=os.environ.get('ADAFORGE_EXPERIMENTAL_NORM_ROPE')
results['protocol']['qkv_prep_environment_override']=os.environ.get('ADAFORGE_EXPERIMENTAL_QKV_PREP')
results['protocol']['prep_cache_environment_override']=os.environ.get('ADAFORGE_EXPERIMENTAL_PREP_CACHE')
results['protocol']['attention_exp_environment_override']=os.environ.get('ADAFORGE_EXPERIMENTAL_ATTENTION_EXP')
results['protocol']['attention_exp_candidate_strategy']='parallel exponentials with sequential FP32 sum, only for visible >= 64'
results['protocol']['gpu_telemetry']=args.gpu_telemetry
results['protocol']['qkv_prep_default_note']='Unset uses each binary default; current builds group preparation for 256/512-channel heads. Set 0 to use the separate path.'
results['protocol']['tiled_gate_default_note']='Unset uses each binary default; defaults may differ between binary revisions.'
path=args.output;path.write_text(json.dumps(results,indent=2),encoding='utf-8')
print('Exact token IDs verified; report: '+str(path),flush=True)

