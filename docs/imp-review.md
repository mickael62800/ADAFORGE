# Relecture d’Imp pour AdaForge / RTX 4090

Référence examinée : `kekzl/imp`, commit `50dd22ba57a4b67c4565eff3ac25fb76ad83fb4a`, copie locale de lecture dans `tools/imp-reference`. Aucun composant Imp n’est intégré ou compilé dans AdaForge. Les principes sont étudiés pour écrire et mesurer nos propres kernels.

## Différences qui changent la priorité

- Le chemin GGUF Q4_0 d’Imp utilise Q8_1 pour les activations et `dp4a` pour les produits entiers. Éviter systématiquement la quantification d’activation n’est donc pas une conclusion à tirer de sa réussite. Source : [DequantTraits Q4_0](https://github.com/kekzl/imp/blob/50dd22ba57a4b67c4565eff3ac25fb76ad83fb4a/src/compute/gemv_dp4a_traits.cuh#L251).
- Imp distingue les calculs par lignes et par partage de la dimension d’entrée entre plusieurs warps d’un bloc. Le choix dépend des dimensions et du nombre de SM, pas seulement du nombre de threads par bloc. Notre balayage précédent du nombre de warps conservait une ligne par warp ; il ne testait donc pas ce partage. Source : [sélection K-parallel](https://github.com/kekzl/imp/blob/50dd22ba57a4b67c4565eff3ac25fb76ad83fb4a/src/compute/gemv_dp4a_traits.cuh#L560).
- Le chemin par lignes d’Imp réutilise des activations Q8_1 en mémoire partagée, avec un pas rembourré pour éviter les conflits de banques. Cette disposition est spécifique à ses accès Q8 ; son pas ne doit pas être recopié tel quel pour nos activations FP32.
- Imp fusionne certaines activations et la quantification Q8_1. L’intérêt serait à réévaluer si nous ajoutons un chemin entier ; le moteur actuel garde ses activations FP32. Source : [activation et quantification](https://github.com/kekzl/imp/blob/50dd22ba57a4b67c4565eff3ac25fb76ad83fb4a/src/compute/gemm_dp4a.cu).
- Les chemins NVFP4 spécialisés `sm_120a` et les résultats sur RTX 5090 ne constituent pas une mesure de notre Gemma Q4_0 sur RTX 4090 `sm_89`. Les graphes CUDA et les buffers résidents sont déjà présents dans AdaForge. Source : [présentation et chemins mesurés](https://github.com/kekzl/imp/tree/50dd22ba57a4b67c4565eff3ac25fb76ad83fb4a).
- Le projet documente lui-même des essais réfutés et des variations de débit entre sessions. Nous conservons aussi les essais sans gain, les IDs exacts et les passages avant/après/avant. Source : [méthodologie](https://github.com/kekzl/imp/blob/50dd22ba57a4b67c4565eff3ac25fb76ad83fb4a/docs/PERF.md).

## Premier essai issu de cette relecture

Un nouveau kernel propre à AdaForge partage une ligne Q4_0 entre quatre warps. Chaque warp calcule une portion distincte de la dimension d’entrée, avec déquantification et accumulation FP32 ; quatre sommes sont ensuite réduites. Les poids du GGUF, les activations FP32 et la tolérance CPU restent inchangés. Ce changement de répartition modifie l’ordre des sommes FP32 et doit donc passer les contrôles numériques complets.

Appliqué seulement à la projection down du FFN (dimension d’entrée 15360), le rapport `benchmarks/optimization-kparallel-down.json` donne 66.87 tok/s avant, 70.82 après et 67.74 au second passage de la version initiale. Les IDs sont identiques. Le contrôle d’une couche complète passe avec un ratio maximal de tolérance 0.02099 (limite 1). Une variante étendue aux autres projections est mesurée séparément avant de choisir la portée finale.

## Suite à évaluer

1. Choisir la portée du partage K selon les mesures end-to-end sur la 4090, puis vérifier les 48 couches et la conversation.
2. Si les kernels FP32 restent limitants, prototyper une quantification d’activation Q8 avec `dp4a`, sans modifier silencieusement le chemin de référence. Mesurer aussi l’erreur, les logits et la qualité sur plusieurs textes : ce serait un nouveau choix de précision, pas une simple réorganisation des mêmes opérations.
3. Construire un vrai chemin par lots pour le prefill et la vérification de propositions avant d’envisager le drafter MTP. Publier séparément les débits avec et sans spéculation.


La variante étendue à toutes les projections donne 72.43 tok/s avant (version down uniquement), 73.64 pour le candidat et 71.94 au second passage. Ce gain additionnel limité à environ 2 % sur le prompt court n’est pas retenu à ce stade. La version finale conserve donc le partage K uniquement dans la projection down du FFN, où le premier gain est plus marqué et où la dimension d’entrée est la plus longue.

Tests ajoutés : dimensions d’entrée 32, 96, 384 et 15360, cinq lignes, échelles négatives, nulles et subnormales, trois amplitudes d’activation, comparaison à la référence CPU FP64. Les tests du workspace, les 25 tests GPU explicites et Clippy passent.

La validation finale des 48 blocs et des logits passe : ratios maximaux 0.29132 (norme finale), 0.42836 (logits bruts) et 0.42835 (softcap), limite 1 avec `atol=rtol=1e-4`.


Comparaison CUDA finale après validation : médiane AdaForge **71.73 tok/s** (70.03–72.52), llama.cpp CUDA **78.52 tok/s** (76.44–80.28). Même prompt, mêmes 24 IDs générés, un échauffement puis cinq répétitions, cache de prompt désactivé et journal 49/49 couches GPU vérifié. Le rapport courant est `benchmarks/comparison-cuda-native.json` ; celui de la première passe est conservé dans `benchmarks/comparison-cuda-native-first-pass.json`. Les scripts de validation de génération et de chat passent après le changement. AdaForge reste sous la référence et sous 100 tok/s.


## Alternatives supplémentaires après le prototype Q8

Cette revue porte toujours sur le commit figé `50dd22ba57a4b67c4565eff3ac25fb76ad83fb4a`. Les pistes ci-dessous sont des candidats, pas des gains mesurés dans AdaForge.

1. **Plusieurs lignes par warp et activations partagées.** Imp dispose de variantes N_ROWS=1/2/4 avec une sélection selon les dimensions, en plus du partage K déjà essayé. Notre prototype Q8 n’utilise encore qu’une ligne par warp et relit les activations depuis la mémoire globale/L1. Étudier ensemble la réutilisation des activations en shared, le pas des banques et 2/4 lignes par warp serait une nouvelle expérience, distincte du gate/up conjoint écarté précédemment. Il faut mesurer la pression sur les registres et le débit complet. Source : [gemv_dp4a_traits.cuh](https://github.com/kekzl/imp/blob/50dd22ba57a4b67c4565eff3ac25fb76ad83fb4a/src/compute/gemv_dp4a_traits.cuh#L793).
2. **Préférence L1/shared par kernel.** Imp demande une préférence max-L1 pour ses GEMV, y compris K-parallel, via cudaFuncSetAttribute. Ce n’est ni le verrouillage du L2 ni une augmentation physique de la mémoire. Candidat peu invasif à tester sur nos kernels Q4 et Q6, à comparer aux choix automatiques du pilote. Source : [gemm_dp4a.cu](https://github.com/kekzl/imp/blob/50dd22ba57a4b67c4565eff3ac25fb76ad83fb4a/src/compute/gemm_dp4a.cu#L643).
3. **Boucle de génération pilotée par le GPU.** Imp contient un CudaGraphConditionalRunner qui enchaîne les tokens avec un état et une condition d’arrêt GPU. AdaForge rejoue les blocs dans un graphe mais lance encore la tête et lit le résultat greedy côté hôte à chaque token. Mesurer les trous de soumission/synchronisation avant de prototyper la tête dans le graphe puis de courtes rafales ou une boucle conditionnelle. La compatibilité précise des nœuds conditionnels, du pilote et de cudarc doit être vérifiée sur la machine ; le protocole doit préserver EOS, budget, historique et streaming. Source : [cuda_graph.cu](https://github.com/kekzl/imp/blob/50dd22ba57a4b67c4565eff3ac25fb76ad83fb4a/src/runtime/cuda_graph.cu#L505).
4. **Spéculation sans modèle supplémentaire.** Le matcher suffix/ngram d’Imp propose des tokens déjà rencontrés dans le prompt ou l’historique ; le modèle vérifie ensuite le lot et les entrées KV rejetées sont annulées. Cela évite le chargement d’un drafter et pourrait servir aux réécritures de code ou textes répétitifs. Il faut d’abord le chemin de vérification par lots ; les bénéfices dépendent fortement du texte. Source : [engine_spec_ngram.cpp](https://github.com/kekzl/imp/blob/50dd22ba57a4b67c4565eff3ac25fb76ad83fb4a/src/runtime/engine_spec_ngram.cpp).

La PDL employée dans certains chemins Imp n’est pas une option pour sm_89 : la documentation NVIDIA exige une capacité de calcul d’au moins 9.0 pour le recouvrement programmatique des kernels. Source : https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html

Priorité proposée : tester d’abord le choix L1/shared (petit changement sans nouvelle précision), puis la réutilisation multi-lignes ; instrumenter ensuite le coût de la boucle CPU avant un chantier de génération autonome. Les techniques NVFP4 sm_120a, les caches d’experts MoE et les résultats de débit multi-utilisateur ne répondent pas directement à notre objectif de décodage d’une conversation Gemma dense sur une 4090.
