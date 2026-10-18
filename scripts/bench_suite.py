"""Speed over a varied prompt set, with and without speculative decoding.

For each prompt: one `adaforge bench` run per mode (warm-up + N measures),
checks that both modes generate exactly the same token IDs, and reports the
median decode speed. Writes benchmarks/suite.json.
"""
import argparse, json, statistics, subprocess, os
from pathlib import Path

PROMPTS = {
    "chat": "Bonjour, comment vas-tu ?",
    "explication": "Explique simplement pourquoi le ciel est bleu.",
    "code": "Écris une fonction Python qui trie une liste de dictionnaires par la clé 'age', avec un exemple.",
    "resume": "Résume ce texte en trois phrases : Le moteur à combustion interne fonctionne en quatre temps. Pendant l'admission, le piston descend et aspire le mélange air-carburant. Pendant la compression, le piston remonte et comprime ce mélange. Pendant la combustion, une étincelle enflamme le mélange, dont l'expansion repousse le piston. Pendant l'échappement, le piston remonte et chasse les gaz brûlés.",
    "traduction": "Traduis en anglais : Le chat dort sur le canapé pendant que la pluie tombe doucement sur les toits de la ville.",
    "correction_code": "Corrige le bug dans ce code et renvoie le code complet corrigé :\ndef moyenne(valeurs):\n    total = 0\n    for v in valeurs:\n        total += v\n    return total / len(valeurs) + 1\n\nprint(moyenne([1, 2, 3]))",
    "recopie": "Recopie exactement ce texte, sans rien changer : Le moteur à combustion interne fonctionne en quatre temps. Pendant l'admission, le piston descend et aspire le mélange air-carburant dans le cylindre. Pendant la compression, le piston remonte et comprime ce mélange.",
}

def run(exe, model, prompt, tokens, runs, speculative):
    env = dict(os.environ)
    env["ADAFORGE_SPECULATIVE"] = "1" if speculative else "0"
    out = subprocess.run([exe, "bench", str(model), prompt, str(tokens), str(runs)],
                         capture_output=True, text=True, encoding="utf-8", env=env, check=True).stdout
    samples = [json.loads(l.removeprefix("BENCH_SAMPLE ")) for l in out.splitlines() if l.startswith("BENCH_SAMPLE ")]
    speeds = [s["decode_tokens"] * 1000.0 / s["decode_ms"] for s in samples if s["decode_ms"] > 0]
    return samples[0]["generated_ids"], statistics.median(speeds), statistics.median(s["ttft_ms"] for s in samples)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("model", type=Path)
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--runs", type=int, default=3)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    exe = str(root / "target/release/adaforge.exe")
    report = {}
    for name, prompt in PROMPTS.items():
        base_ids, base_tps, ttft = run(exe, args.model, prompt, args.tokens, args.runs, False)
        spec_ids, spec_tps, _ = run(exe, args.model, prompt, args.tokens, args.runs, True)
        assert base_ids == spec_ids, f"{name}: speculative output differs"
        report[name] = {"tokens": len(base_ids), "plain_tps": round(base_tps, 2),
                        "speculative_tps": round(spec_tps, 2), "ttft_ms": round(ttft, 1)}
        print(f"{name:16} {len(base_ids):4} tokens  normal {base_tps:7.2f} tok/s  spéculatif {spec_tps:7.2f} tok/s  1er token {ttft:6.1f} ms", flush=True)
    plain = statistics.mean(r["plain_tps"] for r in report.values())
    spec = statistics.mean(r["speculative_tps"] for r in report.values())
    report["_mean"] = {"plain_tps": round(plain, 2), "speculative_tps": round(spec, 2)}
    print(f"{'moyenne':16}             normal {plain:7.2f} tok/s  spéculatif {spec:7.2f} tok/s")
    (root / "benchmarks/suite.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

if __name__ == "__main__":
    main()
