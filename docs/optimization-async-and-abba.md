# Pipeline asynchrone Q4 et qualification ABBA

## Résultat retenu

La comparaison renforcée qualifie le kernel **tuilé synchrone**, sans duplication des poids, sur le prompt d'explication GPU avec 48 tokens de sortie :

| Série | Médiane tokens/s |
|---|---:|
| Référence A1 | 69,7367 |
| Tuiles B1 | 72,2675 |
| Tuiles B2 | 72,4473 |
| Référence A2 | 69,8093 |

Moyenne des médianes : 69,7730 contre 72,3574, soit +2,5844 tok/s (+3,704 %). Dérive des médianes de contrôle : 0,104 %. Les observations individuelles restent dispersées et leurs plages se chevauchent ; il ne s'agit pas d'une preuve de significativité ni d'un débit garanti pour tout prompt. Rapport brut : `benchmarks/optimization-tiled-gate-abba.json`.

Chaque processus effectue un échauffement intégré, trois requêtes supplémentaires conservées dans le rapport mais exclues des statistiques, puis sept requêtes retenues. Le KV est réinitialisé entre requêtes. Toutes les sorties retenues ont exactement les mêmes IDs. Les moteurs s'exécutent séquentiellement. ABBA équilibre une dérive linéaire, sans supprimer une dérive non linéaire ou une activité extérieure.

Le kernel est désormais **sélectionné par défaut uniquement pour width=3840 et intermediate=15360**, la géométrie FFN de notre Gemma. Le chemin natif reste disponible avec `ADAFORGE_EXPERIMENTAL_TILED_GATE=0` ; `1` force le tuilé pour les essais. Cette variable historique sert maintenant d'override. Le prototype Q8 conserve sa propre priorité et ses poids natifs. Le volume des poids gate/up GPU reste égal au natif.

La politique repose aussi sur les validations des 48 blocs/logits, les 27 tests GPU et les comparaisons antérieures sur trois prompts. Le benchmark général de contextes et le profil decode-only des deux moteurs restent à poursuivre ; la cible 80 puis 100 tok/s n'est pas atteinte par cet essai.

Après activation par défaut, les 27 tests GPU, les trois tests du banc de mesure, clippy et le formatage passent. `verify_generation.py` confirme sans override la réponse complète identique à llama.cpp CUDA b11669, l'arrêt de modèle, le préfixe limité et les limites invalides.

## Essais asynchrones écartés

Deux variantes à deux buffers ont été implémentées : un groupe de poids par warp/étape, puis quatre groupes par étape. Les codes passent par des copies `cp.async` de 16 octets, les paires d'échelles par copies de 4 octets ; commits, attentes et synchronisations warp protègent la réutilisation des buffers. La réduction FP32 native est conservée.

| Gate, signal structuré | Tuiles synchrones | Async 1 groupe | Async 4 groupes |
|---|---:|---:|---:|
| Répétition directe | 31,81 µs | 48,08 µs | 36,86 µs |
| Après passe d'éviction 128 Mio | 64,54 µs | 73,73 µs | 74,75 µs |

Les deux variantes perdent également sur les formes down et Q testées. L'oracle CPU et l'identité bit à bit face au kernel natif Q/gate passent, ainsi que les cas partiels 1/5/9 lignes et K=96. Le coût des copies et synchronisations est une explication possible ; aucune attribution de stalls ou de bande passante mesurée n'est revendiquée.

Les sources asynchrones sont isolées dans `kernels/ada/q4_tiled_async_probe.cu`, incluses uniquement par le banc `tiled_probe`. Elles ne sont pas chargées par le moteur. Résultats : `benchmarks/tiled-async-probe.json` (36 cas, cinq variantes alternées, vingt mesures retenues par variante).

## Banc de comparaison

`scripts/compare_optimization.py` accepte maintenant `--abba` et `--discard-leading`. Le rapport garde toutes les observations exclues et retenues, les médianes par série, la dérive du contrôle et l'écart descriptif. Les trois tests sans GPU de `scripts/test_compare_optimization.py` vérifient le décompte, le rejet d'IDs divergents et les limites de requêtes.

Pour reproduire avec le binaire natif archivé :

```powershell
python scripts/compare_optimization.py --baseline tools/optimization-tiled-gate/baseline.exe --output benchmarks/tiled-abba-repeat.json --prompt "Explique simplement pourquoi une carte graphique est utile pour faire fonctionner un modele de langage en local." --tokens 48 --runs 7 --discard-leading 3 --abba
```

Les nouveaux rapports conservent la valeur brute de l'override tiled. Une valeur absente utilise le défaut propre à chaque révision du binaire, elle ne signifie donc plus que les tuiles sont désactivées.
