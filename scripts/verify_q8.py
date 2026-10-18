"""Functional checks for experimental Q8 graphs; does not establish model quality."""
import argparse,json,os,re,subprocess
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('model',type=Path);args=p.parse_args()
root=Path(__file__).resolve().parents[1];results={}
transcript='Mon prénom est Luc.\nQuel est mon prénom ?\n/reset\nBonjour, comment vas-tu ?\n/quit\n'
for mode in ['graph','direct']:
 env=os.environ.copy();env['ADAFORGE_EXPERIMENTAL_Q8']='1';env.pop('ADAFORGE_DISABLE_GRAPHS',None)
 if mode=='direct':env['ADAFORGE_DISABLE_GRAPHS']='1'
 r=subprocess.run([str(root/'target/release/adaforge.exe'),'chat',str(args.model.resolve()),'64'],input=transcript,capture_output=True,encoding='utf-8',env=env,check=True)
 answers=re.findall(r'\nAssistant : (.*?)(?=\nVous >|\Z)',r.stdout,re.S)
 answers=[a.split('\n[Mesures',1)[0].rstrip()for a in answers]
 assert len(answers)==3 and 'Luc' in answers[1] and 'Historique effacé' in r.stdout,r.stdout
 results[mode]=answers
 print('Experimental Q8',mode,'history/reset verified.',flush=True)
assert results['graph']==results['direct'],results
results['scope']='graph/direct equality and history/reset on one transcript; quality not established'
path=root/'benchmarks/q8-functional.json';path.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
print('Graph/direct responses match exactly; report:',path,flush=True)
