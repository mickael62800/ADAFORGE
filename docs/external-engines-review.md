# Analyse de trois moteurs CUDA externes

Cette note analyse des sources externes comme références techniques. Aucun code de ces dépôts n'est intégré à AdaForge. Toute reprise textuelle de code exigerait une revue de licence et une attribution adaptée ; les pistes ci-dessous décrivent des idées à réimplémenter et à valider indépendamment.

## Périmètre et révisions consultées

| Projet | Révision locale | Licence | Cible principale | Rapport avec AdaForge |
|---|---|---|---|---|
| NInfer-4090 | `04266a0fd8446d8c1c3300606c0c60fd9550a4a4` | Apache-2.0 | Qwen3.8-27B, RTX 4090 `sm_89`, Linux | Le plus proche pour les schedules CUDA spécialisés et le MTP, mais poids, modèle et précision différents |
| ExLlamaV2 | `7dc12af3a81f34ac3f27cd7602ed539b638933ca` | MIT | modèles EXL2/GPTQ sur GPU grand public | Bon exemple de préparation hors ligne des poids et de sélection par forme ; son format n'est pas GGUF Q4_0 |
| Redline | `8d2d3e0744ee89d71437dfc771839b0630430494` | MIT | Qwen2.5-1.5B FP16, RTX 4090, service multi-requêtes | Référence de méthodologie et d'attention GQA paginée ; son débit agrégé ne mesure pas notre chat unique |

Les clones sont volontairement placés sous `tools/` et ignorés par le projet : `tools/ninfer-4090-reference`, `tools/exllamav2-reference` et `tools/redline-reference`.

## NInfer-4090

Le projet sépare explicitement plusieurs schedules Q4. Son GEMV `q4_rowsplit_gemv` a notamment une variante **quatre lignes, un warp par ligne** et une variante **une ligne, huit warps**. Il charge des tuiles de codes et de paires d'échelles alignées sur 16 octets avec `cp.async`, puis les consomme depuis la mémoire partagée. Les choix sont des constantes de compilation ; ils ne sont pas décidés dans la boucle de génération.

Ce point est directement applicable, avec adaptation : notre Q4_0 possède des blocs GGUF de 18 octets et des activations FP32, alors que NInfer emploie des poids Q4 rangés séparément et des activations BF16. Il faut donc commencer par un rangement de tuiles bit-à-bit réversible, puis conserver l'accumulation FP32 d'AdaForge. Le `cp.async` de NInfer est une hypothèse mesurable, pas une preuve qu'il améliorera notre GEMV actuel.

NInfer propose aussi un MTP de profondeur 1 à 5, des caches KV propres au drafter et une vérification qui valide le préfixe accepté puis avance l'état. C'est une confirmation pratique du découpage déjà prévu dans M16–M19 : préfill par lots, vérification parallèle, mise à jour ou annulation logique des états KV. Les 148,6 tok/s publiés par le dépôt sont obtenus avec Qwen, MTP3 et 81 % d'acceptation sur du code ; ils ne sont pas un objectif de débit brut Gemma sans spéculation.

Enfin, ses variantes de KV INT8 et E8 4-bit visent surtout les très longs contextes. Elles réduisent les lectures de cache lorsqu'elles deviennent dominantes, au prix d'une nouvelle erreur numérique. Pour AdaForge, elles ne passent devant le travail sur les poids Q4 que lorsque le profil montrera que l'attention de contexte est le goulot.

Sources locales : `src/ops/linear/q4/q4_rowsplit_gemv.cuh`, `src/core/decode_graph.cpp`, `src/ops/kernel/mtp_round.cuh`, `README.md` et `docs/llamacpp-comparison.md`.

## ExLlamaV2

ExLlamaV2 transforme le rangement des poids une fois au chargement : ses poids quantifiés sont mélangés pour que les threads lisent le format attendu par les kernels. Il mémorise aussi des résultats d'autotuning par triplet de forme `(M, K, N)` et utilise une solution de repli quand les mesures sont trop proches ou instables.

L'idée transposable est claire : AdaForge peut qualifier hors chemin critique un petit nombre de kernels adaptés aux formes fixes de Gemma (3840, 4096, 2048, 15360, 262144), et garder un tableau de dispatch déterministe. Cela vaut mieux qu'un autotune à chaque démarrage ou qu'une seule géométrie imposée à toutes les projections.

Le rangement exact d'ExLlamaV2 n'est pas réutilisable : EXL2/GPTQ possède d'autres groupes, permutations et échelles. Il ne faut donc ni copier son layout ni annoncer un gain depuis ses benchmarks. La piste AdaForge est une conversion interne strictement réversible du Q4_0 GGUF, validée octet pour octet avant tout benchmark.

Sources locales : `exllamav2/exllamav2_ext/cuda/q_matrix.cu` et `exllamav2/exllamav2_ext/cuda/q_gemm_autotune.cuh`.

## Redline

Redline montre une attention de décodage GQA bien structurée : un CTA traite une tête KV et le groupe de têtes Q associé, garde l'accumulation du résultat en registres, applique le softmax en ligne et double-bufferise les tuiles K/V avec `cp.async`. Sa conception est une référence utile pour nos couches globales à une tête KV et les contextes longs.

Pour le prompt court à une conversation, Redline n'est pas une source de gain immédiat : sa pagination du KV, ses buckets de graphes et son ordonnanceur sont conçus pour 64 requêtes concurrentes. Ils ajoutent de l'indirection dont notre cache linéaire n'a pas besoin aujourd'hui. Nous les gardons pour M25 et le contexte long, séparés du score de décodage actuel.

Sa contribution immédiatement utile est méthodologique : il publie les entrées, les durées brutes, les conditions de comparaison, les profils Nsight au niveau des noeuds de graphe et les cas où il perd. AdaForge doit adopter ce niveau de traçabilité avant d'accepter un gain : même nombre de tokens, même prompt, même cache, trois séries avant/après/avant et profil distinct du chemin GPU et des attentes hôte.

Sources locales : `src/kernels/paged_attention.cu`, `docs/PROFILING.md`, `docs/DESIGN.md` et `bench/FAIRNESS.md`.

## Décisions pour AdaForge

1. Ajouter un sélecteur de schedule Q4 limité aux formes Gemma, choisi après une qualification stable hors génération.
2. Faire du prototype de tuiles Q4_0 + échelles alignées et `cp.async` une variante FP32 séparée ; elle doit conserver les sorties de référence.
3. Étendre le plan d'attention GQA aux couches globales et longs contextes, sans remplacer le cache linéaire du chat mono-conversation avant mesure.
4. Faire du MTP Gemma un chantier après le préfill par lots et la vérification KV, jamais une comparaison mélangée au débit sans spéculation.
5. Élever le contrat de benchmark et les profils node-level au rang de condition d'adoption d'une optimisation.
