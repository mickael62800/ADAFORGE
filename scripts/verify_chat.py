"""Verify persistent chat, context refusal and reset against local reference."""
import argparse
from pathlib import Path
import re
import subprocess
import tempfile
parser=argparse.ArgumentParser()
parser.add_argument('model',type=Path)
parser.add_argument('--reference',default='llama-completion')
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
exe=str(root/'target/release/adaforge.exe')
first='Mon prénom est Luc.'
second='Quel est mon prénom ?'
lines=first+'\n'+('bonjour '*600)+'\n'+second+'\n/reset\nBonjour, comment vas-tu ?\n/quit\n'
result=subprocess.run([exe,'chat',str(args.model),'64'],input=lines,capture_output=True,encoding='utf-8',check=True)
answers=re.findall(r'\nAssistant : (.*?)(?=\nVous >|\Z)',result.stdout,re.S)
assert len(answers)==3,repr(result.stdout)
answers=[a.split("\n[Mesures",1)[0].rstrip() for a in answers]
assert 'Contexte de 512 tokens dépassé' in result.stdout
assert 'Historique effacé' in result.stdout
assert 'Luc' in answers[1]
assert answers[2]=="Bonjour ! Je vais très bien, merci de demander. Et vous, comment allez-vous aujourd'hui ? Comment puis-je vous aider ?"
def turn(text):
    return '<|turn>user\n'+text+'<turn|>\n<|turn>model\n<|channel>thought\n<channel|>'
# Recompute the exact prior token transcript from scratch in the reference engine.
prompt=turn(first)+answers[0]+'<turn|>\n'+turn(second)
with tempfile.TemporaryDirectory(prefix='adaforge-chat-') as directory:
    path=Path(directory)/'prompt.txt';path.write_bytes(prompt.encode('utf-8'))
    reference=subprocess.run([args.reference,'-m',str(args.model),'-f',str(path),'--no-escape','-n','64','--temp','0','-ngl','99','--no-conversation','--no-display-prompt','--simple-io'],capture_output=True,encoding='utf-8',check=True)
expected=reference.stdout.strip().removesuffix(' [end of text]')
assert answers[1]==expected,repr({'cached':answers[1],'reference':expected})
print('Persistent second turn matches reference full prefill; overflow preserves history; reset verified.')
