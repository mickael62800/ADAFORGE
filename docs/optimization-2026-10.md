# Optimisations d'octobre 2026 : génération, prefill, contexte long

Toutes les optimisations ci-dessous gardent des résultats **bit-à-bit identiques** au chemin précédent : même ordre d'accumulation FP32, mêmes expressions FMA, même arbre de réduction. Chacune a un test GPU dédié qui compare les bits, et les IDs générés par `bench` restent ceux de `benchmarks/comparison-cuda-native.json`. Les scripts `verify_generation.py` et `verify_chat.py` passent.

## Résultats (RTX 4090, gemma-4-12B-it-QAT-Q4_0)

| Mesure | Avant | Après |
|---|---|---|
| Génération, prompt court (« Bonjour, comment vas-tu ? ») | 79 tok/s | ~95 tok/s |
| Génération, réponse qui recopie le prompt (spéculation) | 86 tok/s | ~200 tok/s |
| Génération, contexte de 7 600 tokens | 26 tok/s | ~83 tok/s |
| Premier token, prompt de 20 tokens | 210 ms | ~40 ms |
| Prefill, 7 620 tokens | 95 s | ~16,5 s |
| Contexte maximal du chat | 512 | 10 240 tokens |

Les débits varient d'environ ±2 % d'une série à l'autre. Les décisions ont été prises sur les temps par kernel de Nsight Systems, pas sur `DECODE_TIMING` seul.

## Ce qui a été fait

- **Tête Q6_K** : nouveau rangement sans perte au chargement (codes par lane en u32, échelles int8 en `uint4`, `d` à part), deux lignes par warp, pas d'écriture du softcap inutilisé. 1,46 → 1,06 ms.
- **Normes** : scales et résiduel lus avant la réduction (ils sont froids en DRAM à chaque token) ; post-norme + résiduel + pré-norme FFN fusionnés, post-norme FFN + résiduel fusionnés. `__fmul_rn` empêche la contraction en FMA qui changerait le dernier bit.
- **Attention scindée** en trois kernels (scores, softmax, valeurs) : scores répartis sur beaucoup de blocs avec les clés partagées entre têtes, somme du softmax séquentielle via mémoire partagée, valeurs alimentées par un pipeline `cp.async` (tuiles de 128 tokens, réservé aux largeurs multiples de 32).
- **Passe par lots** (prefill et vérification spéculative) : matvec Q4 et gate en tuiles lisant les poids une fois pour plusieurs tokens (x réordonné pour une lecture `float4` par token), préparation Q/K/V et écriture du cache groupées, attention des n requêtes en un lancement (`blockIdx.z`). Lots de 32 ; gate et down découpés par 8 pour éviter le débordement de registres ; gate à deux tuiles par bloc.
- **Décodage spéculatif greedy** : brouillons par recherche de 3-grammes dans l'historique (6 tokens), vérification par la passe par lots et une tête Q6_K par lots, retour en arrière du cache KV. Les tokens acceptés sont exactement les choix greedy.

## Essais écartés (mesurés)

- Activations Q8 + `dp4a`, W4A16 : aucun gain, le décodage est limité par la lecture des poids (les gros matvec lisent déjà à ~90 % du plafond mesuré de 845 Go/s).
- Rangement Q4 « thread-major » à lectures de 16 octets, préchargement explicite dans `q4_row` et dans gate : neutre ou plus lent (le compilateur pipeline déjà).
- Deux lignes par warp pour les matvec Q4 du décodage, gate à deux tuiles en décodage : neutre ou plus lent.
- Brouillons à partir de 2-grammes : jamais acceptés sur du texte libre.
- Gate en mémoire partagée, gate « une ligne par warp », down à deux lignes par bloc, gate à 4 tuiles : exacts mais pas plus rapides.

## Variables d'environnement

| Variable | Effet |
|---|---|
| `ADAFORGE_SPECULATIVE=0` | désactive le décodage spéculatif |
| `ADAFORGE_SPEC_DEPTH=n` | longueur des brouillons (défaut 6, maximum 7) |
| `ADAFORGE_BATCH_PREFILL=0` | prefill token par token |
| `ADAFORGE_BATCH_VERIFY=1` | compare le prefill par lots au prefill séquentiel (état caché + cache KV, bit à bit) |
| `ADAFORGE_SPLIT_ATTENTION=0` | ancien kernel d'attention unique |
| `ADAFORGE_FUSED_RMS=0` | normes et résiduels non fusionnés |
| `ADAFORGE_Q6_PACKED=0` | tête Q6_K au format GGUF d'origine |
| `ADAFORGE_TIMING_DECODE=1` | temps GPU par token : tête et 48 blocs |
| `ADAFORGE_DIAGNOSTICS=1` | affiche aussi les compteurs de spéculation |

## Limites

- Les couches locales gardent un cache KV pour tout le contexte alors qu'elles ne voient que 1024 tokens : environ 6,7 Go de VRAM qu'un cache circulaire économiserait.
- En conversation libre, la spéculation n'apporte rien sans un petit modèle de brouillon de la même famille ; aucun n'est présent localement.
- Le prefill reste en FP32 sur les unités FMA (environ 450 tokens/s à 7 600 tokens) : les Tensor Cores changeraient l'ordre des additions.
