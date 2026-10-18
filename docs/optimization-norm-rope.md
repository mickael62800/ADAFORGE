# Fusion RMSNorm + RoPE — 18 octobre 2026

## Implémentation

Un bloc CUDA par tête normalise Q ou K puis effectue sa rotation NeoX. La réduction FP32 garde le même ordre que le kernel RMSNorm existant. Un tampon FP32 en mémoire partagée conserve la frontière d'arrondi avant rotation. Les fréquences et la trigonométrie FP64 existantes sont conservées, avec conversion FP32 identique avant rotation. V conserve son traitement séparé, y compris dans les blocs globaux à projection K/V partagée.

Le chemin couvre les têtes de 256 et 512 canaux, positions statiques et positions dynamiques des graphes CUDA. Deux lancements sont supprimés par couche et par étape, soit 96 pour 48 couches. Aucun changement de quantification, de cache KV ou de précision des activations.

`ADAFORGE_EXPERIMENTAL_NORM_ROPE=1` active la variante ; `0` permet le chemin séparé. Le log `ADAFORGE_CONFIG` indique le nombre de couches fusionnées. Les autres expériences Q6 et QKV restent désactivées.

## Validation

- Comparaison bit pour bit des kernels fusionné et séparés : largeurs 256/512, trois têtes, échelles signées, positions 0, 1, 127, 4095, 131071 et 262143, chemins statique et dynamique.
- Graphe CUDA contre exécution directe : tokens et positions variables, caches et reset vérifiés.
- Référence CPU indépendante sur 48 blocs et 262144 logits : ratio maximal 0,42836 (limite 1, atol=rtol=1e-4).
- Compilation release, formatage et clippy sans avertissement réussis.

Résultats : `benchmarks/optimization-norm-rope-validation.json`.

## Profilage

Deux captures decode seul, 24 projections finales et 23 intervalles entre projections, excluant chargement et prefill. La somme des durées RMSNorm/RoPE par intervalle passe de 1,5790 à 1,4719 ms, soit environ 0,1071 ms de moins. Le compte de kernels diminue de 96 par intervalle. Cette famille comprend aussi les normalisations qui ne changent pas. Les captures datent de moments différents : il s'agit d'une observation descriptive, pas d'une mesure de débit contrôlée.

Sources : `benchmarks/profile-ada-decode-analysis.json`, `benchmarks/profile-norm-rope-analysis.json`, `benchmarks/optimization-norm-rope-profile-summary.json`.

## Débit non profilé

ABBA, cinq mesures conservées par processus, un échauffement interne puis une requête supplémentaire écartée, identifiants de prompt et de sortie identiques.

Sur « Bonjour, comment vas-tu ? », 24 tokens : médianes de référence 76,985 puis 76,412 ; variante 77,581 puis 77,282 tokens/s. Moyenne des médianes : +0,956 %, dérive des contrôles -0,745 %. Ce résultat seul reste modeste.

Données brutes : `benchmarks/optimization-norm-rope-abba.json`. Binaire de référence : `tools/optimization-norm-rope/baseline.exe`.

Sur « Compte de 1 à 100, un nombre par ligne, sans introduction. », 64 tokens : référence 69.964, variante 70.822 tokens/s (moyennes des médianes), soit 1.228 %. Dérive des contrôles : 2.493 %. Résultats bruts : `benchmarks/optimization-norm-rope-counting-abba.json`.

## Décision

Variante numériquement validée, réduction du coût local observée et deux comparaisons de débit positives. Le gain global reste voisin de 1 % et les contrôles varient de façon comparable ou supérieure : pas de gain reproductible suffisamment établi pour activation par défaut. La fusion reste expérimentale et désactivée. La suite consiste à fusionner plus largement la préparation Q/K/V afin de réduire davantage les lancements, puis refaire le même protocole. Les gains observés ici ne se rajoutent pas mécaniquement à ceux d’une future fusion.
