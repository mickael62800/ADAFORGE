# Attention : exponentielles parallèles — 18 octobre 2026

## Pourquoi et comment

Le kernel de référence fait calculer le softmax par un seul thread. La variante parallélise les exponentielles et les divisions, tout en conservant le maximum séquentiel et surtout la somme des exponentielles dans l'ordre FP32 original. Les scores QK et l'accumulation des valeurs ne changent pas. Des synchronisations supplémentaires rendent les écritures partagées visibles.

La première version appliquait ce traitement à toutes les longueurs. La version actuelle le réserve aux séquences visibles d'au moins 64 scores ; en dessous, elle conserve l'algorithme séquentiel. Ce seuil est une hypothèse de travail, pas un optimum démontré par balayage. Même lorsque le petit chemin est exécuté, l'image du kernel et ses ressources compilées peuvent différer : cela ne garantit pas une durée identique.

Activation : `ADAFORGE_EXPERIMENTAL_ATTENTION_EXP=1`. Variante désactivée par défaut. Les logs `ADAFORGE_CONFIG` indiquent `parallel_exp_layers` et `parallel_exp_min_visible`. Aucun changement des poids, du cache KV ou des calculs FP32. La préparation Q/K/V groupée déjà qualifiée reste le défaut ; la fusion d'écriture cache est désactivée.

## Justesse

Test bit pour bit contre le kernel séquentiel : têtes 256 et 512, longueurs 1, 3, 31, 63, 64, 65, 127, 512 et 1025, fenêtres 0 et 64, Q/K non nuls, sorties statiques et dynamiques. Tests CPU de masque causal, fenêtre locale et regroupement des têtes, stabilité avec scores extrêmes, puis graphe contre exécution directe avec changements de tokens, positions et reset : réussis. La génération des benchmarks conserve tous les identifiants de prompt et de sortie.

## Mesures sans profiler

ABBA, cinq mesures conservées par processus, échauffement interne plus une requête mesurée écartée. Chaque résultat ci-dessous est une moyenne des deux médianes de séries, pas une moyenne de tous les tokens.

| Variante / test | Référence tokens/s | Variante tokens/s | Delta descriptif | Dérive des contrôles |
|---|---:|---:|---:|---:|
| Initiale sans seuil, 24 tokens | 75.375 | 74.905 | -0.624 % | -4.258 % |
| Adaptative, comptage 128 tokens | 65.004 | 66.953 | 2.998 % | 1.314 % |
| Adaptative, Bonjour 24 tokens | 76.538 | 75.830 | -0.924 % | -0.907 % |

Les trois fichiers JSON correspondants sont conservés dans `benchmarks/`. Référence archivée : `tools/optimization-attention-exp/baseline.exe`.

## Décision et portée

Le test long indique environ +3 % (+1,95 token/s) avec deux médianes candidates au-dessus des deux contrôles. Le court adaptatif indique environ -0,92 %, avec dérive des contrôles comparable. Ces observations ne démontrent ni un gain universel ni une régression statistiquement établie. Elles ne justifient pas l'activation globale : variante conservée pour les recherches, désactivée par défaut. Il faudra mesurer un balayage de longueurs et le coût isolé du kernel, ou choisir des graphes distincts pour conserver le kernel natif aux petites positions. Aucune modification de précision n'est nécessaire à cette piste.

## Télémétrie

`python scripts/compare_optimization.py --gpu-telemetry ...` ajoute des relevés nvidia-smi avant/après chaque processus : température, puissance, fréquences SM/mémoire, utilisation et VRAM. Les appels sont extérieurs aux requêtes chronométrées. Ce sont des instantanés, pas une trace continue ; ils ne permettent pas d'attribuer à eux seuls une variation de débit à la température ou aux fréquences. Les compteurs mémoire matériels ne sont pas mesurés.

La comptabilité des requêtes avec télémétrie est testée : le nombre de séries/mesures, les échantillons écartés et le calcul du delta restent inchangés. L'option est facultative et désactivée par défaut.

Vérification finale du chemin par défaut : 31 tests GPU réussis. Quatre tests de comptabilité des benchmarks réussis. Les options expérimentales restent désactivées.
