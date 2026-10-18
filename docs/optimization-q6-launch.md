# Essai de géométrie Q6 — 18 octobre 2026

La projection finale identifiée par Nsight (~1,55 ms de médiane du kernel AdaForge contre ~1,00 ms pour la projection Q6 de llama-server dans les captures) constitue une cible mesurée. Ces durées profilées ne sont pas des benchmarks de débit et les chemins de précision des moteurs diffèrent.

## Hypothèse testée

Regrouper huit lignes par bloc CUDA au lieu de quatre pour diminuer le nombre de blocs de la projection finale. Un warp par ligne reste inchangé : mêmes poids Q6_K, activations FP32, ordre des opérations et réduction par ligne. Aucun changement de précision.

Le kernel déduit maintenant le nombre de lignes par bloc de blockDim. Le réglage par défaut reste quatre lignes. Le réglage expérimental `ADAFORGE_EXPERIMENTAL_Q6_ROWS=8` active huit lignes, dans le diagnostic et la tête résidente. La session mémorise ce réglage à sa création.

## Validation et décision

Test GPU Q6 signé, sous-normal, plusieurs blocs et groupe final incomplet : réussi. Compilation release et clippy sans avertissement : réussis. Les identifiants de prompt et de sortie sont identiques sur toutes les séries de génération.

Benchmark non profilé ABBA, 24 tokens, cinq mesures conservées par processus, un échauffement interne et une requête mesurée supplémentaire écartée :

| Série | Tokens/s, médiane |
|---|---:|
| Référence avant | 75,875 |
| Variante | 70,873 |
| Variante répétée | 74,540 |
| Référence après | 74,434 |

Moyenne des médianes : référence 75,154 ; variante 72,706, soit -3,26 %. Dérive des contrôles : -1,90 %. La variante reste désactivée. Ces chiffres concernent ce prompt court et ne constituent pas un débit général garanti.

Données brutes : `benchmarks/optimization-q6-rows8-abba.json`. Binaire de référence : `tools/optimization-q6-launch/baseline.exe`.

La suite utile pour cette tête est de mesurer séparément les coûts de lecture et déquantification, puis de tester un rangement mémoire différent ou une fusion de sélection. Augmenter simplement la taille des blocs ne suffit pas. Les optimisations RMSNorm/RoPE et attention restent d'autres cibles indiquées par les traces.
