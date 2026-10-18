"""Compare our tokenizer to local llama-tokenize. No weights are downloaded."""
import argparse, json, subprocess, tempfile
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument("model",type=Path)
parser.add_argument("--reference",default="llama-tokenize")
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
texts=["", "Bonjour, comment vas-tu ?", "Hello world!", "é è à ç œ É", "e\u0301 café", "👋🌍🙂", "你好，世界！", "مرحبا بالعالم", "Привет мир", "  espaces   répétés  ", "\n", "\n\n", "\n"*40, "a\nb\n", "a\r\nb", "\tindent\t\tfin", "def hello(x):\n    return x + 42\n", "1234567890 3.14159 -42", "<bos>Bonjour<eos>", "<|turn>user\nBonjour<turn|>", "<|tool_response>ok<tool_response|>", "<|channel>analysis<channel|>", "rare: \U00010348 \U0001FAE8", "▁ a▁b", "a"*127, "abababababababab", "\x00 fin", "</s>abc", r"literal \n and \t"]
cases=[]
for text in texts:
    for bos,parse in [(True,False),(False,False),(True,True)]:
        cases.append((text,bos,parse))
fixtures=[]
with tempfile.TemporaryDirectory(prefix="adaforge-tokenizer-") as directory:
    prompt=Path(directory)/"prompt.txt"
    for text,bos,parse in cases:
        # Binary file input preserves CRLF and NUL on Windows; no escape expansion.
        prompt.write_bytes(text.encode("utf-8"))
        ref=[args.reference,"-m",str(args.model),"-f",str(prompt),"--no-escape","--ids"]
        if not bos:ref.append("--no-bos")
        if not parse:ref.append("--no-parse-special")
        reference=subprocess.run(ref,capture_output=True,check=True)
        expected=json.loads(reference.stdout.decode("utf-8").strip())
        fixtures.append(f"{int(bos)}\t{int(parse)}\t{text.encode('utf-8').hex()}\t{','.join(map(str,expected))}")
path=root/"crates/model-formats/tests/fixtures/gemma4_tokenizer.tsv"
path.write_text("# Reference: llama.cpp build 10869 (30b6a755e); Gemma 4 12B IT QAT Q4_0\n# Columns: add_bos, parse_special, UTF8 hex, token IDs\n"+"\n".join(fixtures)+"\n",encoding="utf-8")
print(f"Reference fixtures: {len(cases)} cases",flush=True)
import os
result=subprocess.run(["cargo","test","-p","adaforge-model-formats","--test","tokenizer_gemma","--","--ignored"],cwd=root,env={**os.environ,"ADAFORGE_TEST_MODEL":str(args.model.resolve())})
raise SystemExit(result.returncode)
