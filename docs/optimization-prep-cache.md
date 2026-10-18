# Écriture KV fusionnée avec la préparation — 18 octobre 2026

La variante expérimentale `ADAFORGE_EXPERIMENTAL_PREP_CACHE=1` écrit K/V directement dans le slot courant du cache depuis la préparation groupée. Elle concerne les graphes CUDA ; le chemin direct conserve ses copies. Le kernel d'écriture séparé disparaît, soit 48 lancements supplémentaires en moins par token. Même format et mêmes opérations FP32, calcul inchangé.

Les tests GPU vérifient les sorties Q/K/V bit pour bit, les slots préfixes/suffixes inchangés, les têtes 256/512 et les positions 0/1/127. Le test de graphe contre exécution directe avec changements de tokens, positions et resets passe avec cette option. Les identifiants de génération sont identiques au binaire de référence dans toutes les séries ABBA.

ABBA court : références [78.70052431658006, 75.52428631191269], variantes [78.46409602367842, 76.53724219012317] tokens/s. Delta descriptif des moyennes des médianes 0.504 %, dérive des contrôles -4.036 %.

Gain global non démontré : variante désactivée par défaut. Données brutes : `benchmarks/optimization-prep-cache-abba.json`. Référence : `tools/optimization-prep-cache/baseline.exe`.
