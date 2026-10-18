"""Compare one-turn greedy AdaForge output to a local llama.cpp reference."""
import argparse
from pathlib import Path
import subprocess
import tempfile

parser=argparse.ArgumentParser()
parser.add_argument('model',type=Path)
parser.add_argument('--reference',default='llama-completion')
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
text='Bonjour, comment vas-tu ?'
prompt='<|turn>user\n'+text+'<turn|>\n<|turn>model\n<|channel>thought\n<channel|>'
with tempfile.TemporaryDirectory(prefix='adaforge-generation-') as directory:
    path=Path(directory)/'prompt.txt'
    path.write_bytes(prompt.encode('utf-8'))
    ref=subprocess.run([args.reference,'-m',str(args.model),'-f',str(path),'--no-escape','-n','64','--temp','0','-ngl','99','--no-conversation','--no-display-prompt','--simple-io'],capture_output=True,encoding='utf-8',check=True)
    # llama-completion appends this UI marker after an EOG token.
    expected=ref.stdout.strip().removesuffix(' [end of text]')
actual=subprocess.run([str(root/'target/release/adaforge.exe'),'ask',str(args.model),text,'64'],capture_output=True,encoding='utf-8',check=True)
answer=actual.stdout.split('Réponse : ',1)[1].rsplit('\nArrêt : ',1)[0]
assert answer==expected,repr({'adaforge':answer,'reference':expected})
assert 'Arrêt : marqueur de fin du modèle' in actual.stdout
print('One-turn greedy completion matches local reference exactly; model stop verified.')

limited=subprocess.run([str(root/'target/release/adaforge.exe'),'ask',str(args.model),text,'2'],capture_output=True,encoding='utf-8',check=True)
partial=limited.stdout.split('Réponse : ',1)[1].rsplit('\nArrêt : ',1)[0]
assert answer.startswith(partial) and partial
assert '2 tokens générés' in limited.stdout and 'Arrêt : limite de tokens' in limited.stdout
for limit in ['0','129','bad']:
    invalid=subprocess.run([str(root/'target/release/adaforge.exe'),'ask','missing.gguf',text,limit],capture_output=True)
    assert invalid.returncode != 0 and b'GGUF' not in invalid.stderr
print('Token limit, deterministic prefix and invalid limits verified.')
