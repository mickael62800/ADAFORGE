"""Local, warmed comparison: AdaForge CUDA vs official llama.cpp CUDA GPU."""
import argparse,json,os,socket,statistics,subprocess,threading,time,urllib.request
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('model',type=Path);p.add_argument('--runs',type=int,default=5);p.add_argument('--native',action='store_true',default=True,help='Use llama.cpp native FP16 KV and auto flash attention');p.add_argument('--fp32',action='store_false',dest='native',help='Match FP32 KV and disable flash attention');p.add_argument('--server',type=Path,default=Path(__file__).resolve().parents[1]/'tools/llama-b11669-cuda12.4/bin/llama-server.exe');args=p.parse_args()
if not 1<=args.runs<=20:raise SystemExit('runs: 1..20')
if os.environ.get('ADAFORGE_EXPERIMENTAL_Q8')=='1':raise SystemExit('Unset ADAFORGE_EXPERIMENTAL_Q8 for the validated FP32 comparison; use compare_q8.py for the experiment.')
root=Path(__file__).resolve().parents[1];out=root/'benchmarks';out.mkdir(exist_ok=True)
message='Bonjour, comment vas-tu ?';tokens=24
reference_version=subprocess.run([str(args.server.resolve()),'--version'],capture_output=True,text=True,check=True).stdout.strip()
flags=getattr(subprocess,'CREATE_NO_WINDOW',0)
def memory():
    r=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,creationflags=flags,timeout=5)
    return int(r.stdout.splitlines()[0])
class Monitor:
    def __enter__(self):
        self.idle=memory();self.peak=self.idle;self.stop=threading.Event()
        def poll():
            while not self.stop.is_set():
                try:self.peak=max(self.peak,memory())
                except (ValueError,subprocess.SubprocessError):pass
                self.stop.wait(.5)
        self.thread=threading.Thread(target=poll);self.thread.start();return self
    def __exit__(self,*_):self.stop.set();self.thread.join()
with Monitor() as am:
    run=subprocess.run([str(root/'target/release/adaforge.exe'),'bench',str(args.model.resolve()),message,str(tokens),str(args.runs)],capture_output=True,text=True,encoding='utf-8',check=True)
ada=[json.loads(line.removeprefix('BENCH_SAMPLE ')) for line in run.stdout.splitlines() if line.startswith('BENCH_SAMPLE ')]
assert len(ada)==args.runs
assert all(x['prompt_ids']==ada[0]['prompt_ids'] and x['generated_ids']==ada[0]['generated_ids'] and len(x['generated_ids'])==tokens for x in ada)
print('AdaForge warmed samples collected.',flush=True)
with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
base=f'http://127.0.0.1:{port}'
suffix='native' if args.native else 'fp32'
logpath=out/f'llama-cuda-{suffix}.log'
command=[str(args.server.resolve()),'-m',str(args.model.resolve()),'-ngl','99','--device','CUDA0','-c','512','-np','1','-ctk',('f16' if args.native else 'f32'),'-ctv',('f16' if args.native else 'f32'),'-fa',('auto' if args.native else 'off'),'--fit','off','--log-verbosity','4','--host','127.0.0.1','--port',str(port)]
def request():
    payload={'prompt':ada[0]['prompt_ids'],'n_predict':tokens,'temperature':0,'top_k':1,'top_p':1,'min_p':0,'repeat_penalty':1,'seed':0,'cache_prompt':False,'return_tokens':True,'stream':True,'timings_per_token':True}
    req=urllib.request.Request(base+'/completion',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    started=time.perf_counter();arrivals=[];ids=[];last={}
    with urllib.request.urlopen(req,timeout=120) as response:
        for raw in response:
            if not raw.startswith(b'data: '):continue
            data=json.loads(raw[6:]);now=time.perf_counter();last=data
            for token in data.get('tokens',[]):ids.append(token);arrivals.append(now)
    assert len(ids)==tokens,(len(ids),last)
    assert ids==ada[0]['generated_ids'],('Different output tokens',ids,ada[0]['generated_ids'])
    timings=last.get('timings',{})
    assert timings.get('prompt_n')==len(ada[0]['prompt_ids']),('Prompt cache reuse or different prompt count',timings)
    return {'generated_ids':ids,'ttft_ms':(arrivals[0]-started)*1000,'decode_tokens':len(ids)-1,'decode_ms':(arrivals[-1]-arrivals[0])*1000,'server_timings':timings}
llama=[]
with Monitor() as lm,logpath.open('w',encoding='utf-8') as logfile:
    server=subprocess.Popen(command,stdout=logfile,stderr=logfile,creationflags=flags)
    try:
        deadline=time.monotonic()+90
        while True:
            if server.poll() is not None:raise RuntimeError('Server failed: '+logpath.read_text(encoding='utf-8')[-6000:])
            try:
                with urllib.request.urlopen(base+'/health',timeout=1) as r:
                    if r.status==200:break
            except (OSError,TimeoutError):pass
            if time.monotonic()>deadline:raise RuntimeError('Server startup timeout')
            time.sleep(.25)
        request()
        for i in range(args.runs):llama.append(request());print(f'llama.cpp CUDA sample {i+1}/{args.runs}',flush=True)
    finally:
        server.terminate()
        try:server.wait(timeout=10)
        except subprocess.TimeoutExpired:server.kill();server.wait()
logs=logpath.read_text(encoding='utf-8')
verified=('offloaded 49/49 layers' in logs and 'CUDA0' in logs)
def summary(samples):
    rates=[s['decode_tokens']*1000/s['decode_ms'] for s in samples]
    return {'decode_tps_median':statistics.median(rates),'decode_tps_min':min(rates),'decode_tps_max':max(rates),'ttft_ms_median':statistics.median(s['ttft_ms'] for s in samples)}
report={'model':str(args.model.resolve()),'runs':args.runs,'gpu_offload_verified':verified,'reference_native':args.native,'prompt_tokens':len(ada[0]['prompt_ids']),'output_tokens':tokens,'adaforge':summary(ada),'llama_cpp_cuda':summary(llama),'adaforge_vram_device_idle_mib':am.idle,'adaforge_vram_device_sampled_peak_mib':am.peak,'llama_vram_device_idle_mib':lm.idle,'llama_vram_device_sampled_peak_mib':lm.peak,'adaforge_samples':ada,'llama_samples':llama,'reference_command':command,'reference_version':reference_version,'notes':['One warm-up; weights stay loaded; KV prefix reset for every request.','Same token IDs, greedy sampling, context 512, one request at a time; reference KV/FA settings in command.','AdaForge TTFT is backend wall time; llama.cpp TTFT includes local HTTP/SSE.','Decode rate: (N-1)/(last token ready minus first token ready); timings exclude model loading.','VRAM sampled for the whole GPU, includes desktop and other applications; not per-process allocation.','AdaForge uses FP32 KV; reference may use native FP16/FA auto (see command and reference_native).']}
path=out/f'comparison-cuda-{suffix}.json';path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:report[k] for k in ['adaforge','llama_cpp_cuda']},indent=2),flush=True)
print('Report: '+str(path),flush=True)

assert verified, 'GPU offload unverified: see '+str(logpath)
