"""Profile an isolated local llama-server CUDA session and retain logs/timings."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--seconds', type=int, default=45)
args = p.parse_args()
if not 20 <= args.seconds <= 180:
    p.error('seconds: 20..180')
out = ROOT / 'benchmarks'
nsys = Path('C:/Program Files/NVIDIA Corporation/Nsight Systems 2026.1.3/target-windows-x64/nsys.exe')
server = ROOT / 'tools/llama-b11669-cuda12.4/bin/llama-server.exe'
model = ROOT / 'models/gemma-4-12b-it-qat-q4_0/gemma-4-12B-it-QAT-Q4_0.gguf'
reference = json.loads((out / 'comparison-cuda-native.json').read_text(encoding='utf-8'))
prompt = reference['adaforge_samples'][0]['prompt_ids']
expected = reference['adaforge_samples'][0]['generated_ids']
with socket.socket() as sock:
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
base = f'http://127.0.0.1:{port}'
target_log = out / f'profile-llama-server-target-{time.time_ns()}.log'
server_args = [str(server), '-m', str(model), '-ngl', '99', '--device', 'CUDA0',
               '-c', '512', '-np', '1', '-ctk', 'f16', '-ctv', 'f16', '-fa', 'auto',
               '--fit', 'off', '--log-verbosity', '4', '--log-file', str(target_log),
               '--host', '127.0.0.1', '--port', str(port)]
command = [str(nsys), 'profile', '--trace=cuda', '--cuda-graph-trace=node',
           '--sample=none', '--cpuctxsw=none', '--force-overwrite=true',
           f'--duration={args.seconds}', '--kill=true', '-o', str(out / 'profile-llama-server'),
           *server_args]
samples = []
flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
logpath = out / 'profile-llama-server.log'
with logpath.open('w', encoding='utf-8') as logfile:
    process = subprocess.Popen(command, stdout=logfile, stderr=logfile, creationflags=flags)
    try:
        deadline = time.monotonic() + args.seconds
        while True:
            if process.poll() is not None:
                raise RuntimeError('Profiler/server exited before health; see log')
            try:
                with urllib.request.urlopen(base + '/health', timeout=1) as response:
                    if response.status == 200:
                        break
            except (OSError, TimeoutError):
                pass
            if time.monotonic() > deadline:
                raise RuntimeError('Server startup timeout')
            time.sleep(.2)
        print('Profiled CUDA server ready.', flush=True)
        for index in range(4):
            payload = {'prompt': prompt, 'n_predict': 24, 'temperature': 0, 'top_k': 1,
                       'top_p': 1, 'min_p': 0, 'repeat_penalty': 1, 'seed': 0,
                       'cache_prompt': False, 'return_tokens': True, 'stream': True,
                       'timings_per_token': True}
            req = urllib.request.Request(base + '/completion', data=json.dumps(payload).encode(),
                                         headers={'Content-Type': 'application/json'})
            started = time.perf_counter()
            ids, arrivals, last = [], [], {}
            with urllib.request.urlopen(req, timeout=30) as response:
                for raw in response:
                    if raw.startswith(b'data: '):
                        last = json.loads(raw[6:])
                        now = time.perf_counter()
                        for token in last.get('tokens', []):
                            ids.append(token)
                            arrivals.append(now)
            assert ids == expected and len(ids) == 24, (ids, expected)
            assert last['timings']['prompt_n'] == len(prompt), last['timings']
            samples.append({'warmup': index == 0, 'prompt_ids': prompt, 'generated_ids': ids,
                            'ttft_ms': (arrivals[0] - started) * 1000,
                            'decode_ms': (arrivals[-1] - arrivals[0]) * 1000,
                            'decode_tokens': len(ids) - 1, 'server_timings': last['timings']})
            print(f'Profiled request {index + 1}/4 completed; exact IDs.', flush=True)
            # Deliberate idle marker to separate HTTP requests in the GPU trace.
            time.sleep(.3)
        process.wait(timeout=args.seconds + 20)
    finally:
        if process.poll() is None:
            # This is the dedicated profiler process and its dedicated child server only.
            subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                           capture_output=True, check=False, creationflags=flags)
            process.wait(timeout=10)
logs = target_log.read_text(encoding='utf-8')
verified = 'offloaded 49/49 layers' in logs and 'CUDA0' in logs
report = {'reference_command': server_args, 'profiler_command': command,
          'samples': samples, 'gpu_offload_verified': verified, 'target_log': str(target_log),
          'scope': 'Traced HTTP/SSE requests, one warmup and three measured requests; not an unprofiled performance benchmark. Full trace includes loading and prefill; analyze the final request decode separately.'}
(out / 'profile-llama-server-requests.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
assert verified, 'GPU placement not verified; request data retained, see target log'
print('Server trace, logs and request timings saved.', flush=True)
