# Écart avec llama.cpp CUDA : première lecture de la version comparée

Sources récupérées dans `tools/llama-source-reference`, tag `b11669`, commit `620d2225cc436ab0b001e3200e590e3b286ef9ab` : même version que la distribution CUDA officielle employée pour nos mesures. Aucune intégration de ces sources dans AdaForge.

## Ce que les sources et le journal établissent

| Point | llama.cpp | AdaForge | Conclusion mesurable |
|---|---|---|---|
| Produits Q4_0 | Chemin MMVQ Q4_0 × Q8_1 avec dp4a, correction du zéro Q4 au niveau d’une somme | Production FP32 ; prototype Q8 limité au FFN, décompression signée par paquets | Le prototype Q8 actuel ne reproduit ni toute la couverture ni toutes les optimisations du chemin llama.cpp |
| Tête Q6_K | Fonction vec_dot_q6_K_q8_1 et dispatch MMVQ disponibles | Q6_K × FP32 | Étudier séparément le chemin de tête ; confirmer par trace le kernel réellement choisi pour cette exécution |
| Répartition des produits | Tables de paramètres, nombre de warps, lignes/bloc, nombre de colonnes de sortie et cas small_k | Quelques configurations fixes, down partagé K | La largeur et le type comptent ; comparer sur chacune de nos formes |
| Norme / RoPE / écriture KV | Fusions conditionnelles RMSNorm × échelle → RoPE, avec éventuellement SET_ROWS | Kernels distincts pour ces étapes | Vérifier les conditions Gemma et les arrondis avant un essai ciblé |
| Cache KV | Journal natif et protocole : FP16, Flash Attention auto | Cache FP32 et attention propre | La comparaison native mesure les choix respectifs ; il faut isoler les effets avant d’attribuer tout l’écart aux projections |
| Traitement du prompt | Backend disposant d’un vrai chemin multi-tokens | Prompt encore traité token par token | Explique une piste pour le TTFT, pas automatiquement tout l’écart de décodage |

Sources : [vecdotq.cuh](https://github.com/ggml-org/llama.cpp/blob/620d2225cc436ab0b001e3200e590e3b286ef9ab/ggml/src/ggml-cuda/vecdotq.cuh), [mmvq.cu](https://github.com/ggml-org/llama.cpp/blob/620d2225cc436ab0b001e3200e590e3b286ef9ab/ggml/src/ggml-cuda/mmvq.cu), [dispatch et fusions](https://github.com/ggml-org/llama.cpp/blob/620d2225cc436ab0b001e3200e590e3b286ef9ab/ggml/src/ggml-cuda/ggml-cuda.cu), [graphe Gemma 4](https://github.com/ggml-org/llama.cpp/blob/620d2225cc436ab0b001e3200e590e3b286ef9ab/src/models/gemma4.cpp). Le journal vérifié est `benchmarks/llama-cuda-native.log`.

## Ce qui n’est pas encore établi

La présence d’un kernel ou d’une fusion dans le code ne prouve pas son activation dans notre requête. `flash_attn=auto` ne suffit pas non plus à désigner le kernel d’attention exécuté. Une trace Nsight du serveur de référence, avec la même charge, est nécessaire pour comparer les kernels effectifs et leurs temps avec AdaForge. Le différentiel d’environ 7–9 tok/s n’a pas encore été décomposé opération par opération.

Le code Q4/Q8 de llama.cpp fait les produits sur les nibbles non signés et corrige ensuite leur décalage de 8 avec une somme d’activations. Notre prototype construit actuellement des paquets INT8 signés avant dp4a. Pré-calculer la somme quantifiée par bloc est donc une expérience nouvelle pour réduire les instructions de ce prototype sans changer les valeurs INT8. Cela ne justifie pas à lui seul un nouveau choix de précision par défaut.

## Priorités supplémentaires à ajouter au catalogue

- Profiler la référence CUDA exacte plutôt que comparer seulement son débit final.
- Comparer le coût des projections Q4, de la tête Q6 et de l’attention à formes égales et avec les choix de précision explicités.
- Tester la correction du zéro Q4 par somme de bloc dans le prototype Q8.
- Étudier la quantification d’activation pour Q/K/V et la tête, séparément du FFN et avec une évaluation de précision.
- Étudier les fusions norme de tête / échelle / RoPE / cache, en validant leur applicabilité exacte à Gemma et leur justesse.

llama.cpp prouve qu’un débit proche de 80 tok/s est atteignable sur cette machine avec ce modèle et ses choix de calcul. Cela ne prouve ni que son code sera optimal pour tous les cas ni que 100 tok/s sera atteint par une optimisation donnée.
