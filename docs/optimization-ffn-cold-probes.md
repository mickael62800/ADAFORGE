# FFN : disposition down et déroulage gate/up — 18 octobre 2026

Objectif : accélérer le FFN RTX 4090 sans modifier les activations FP32 ni l'ordre des réductions.

## Expériences

- `ffn_down_probe` : quatre variantes de déroulage (1/2/4/8) d'une disposition entrelacée de quatre lignes, 16 warps par groupe. Comparaison avec `q4_matvec_kparallel`.
- `ffn_gate_probe` : quatre variantes de déroulage de la projection entrelacée existante, sur la concaténation des matrices gate et up. Comparaison avec `q4_tiled_split`, même corps de calcul que le kernel gate de production, sortie contiguë au lieu de deux pointeurs.
- Poids réels des blocs 0 et 5, trois familles d'entrées et matrices synthétiques de 1/5/9 lignes. Vérification inverse du packing, égalité bit à bit avec le kernel de référence et référence CPU FP64 à atol=rtol=1e-4 : réussies.
- 20 mesures par variante après deux passages, ordre tournant. Régimes répété et après écriture d'un tampon d'éviction de 128 Mio. L'éviction est une approximation d'un flux de poids ; elle ne reproduit pas une génération complète. Pas de mesure des compteurs DRAM.

## Résultats et décision

| Projection | Référence, poids répétés | Variante favorable, poids répétés | Référence après éviction | Variante après éviction |
|---|---:|---:|---:|---:|
| Down | ~30–32 µs | ~25–26 µs | ~65,5–66 µs | ~68,6 µs |
| Gate + up | ~44–46 µs | ~43–45 µs | ~122,9 µs | ~122,9–124,4 µs |

Aucun gain à froid convaincant. Ces kernels restent des expériences, sans modification du chemin de production. Aucun nouveau gain en tokens/s n'est revendiqué ; pas de benchmark complet lancé pour ces candidats défavorables. Les résultats chauds seuls auraient conduit à une conclusion trompeuse, surtout pour down.

Traces : `benchmarks/ffn-down-tiled-probe.txt` et `benchmarks/ffn-gate-unrolled-probe.txt`.

Prochaine hypothèse à examiner : réutilisation des activations et organisation des accès mémoire à travers plusieurs groupes de lignes, avec mesure à froid dès le départ. Le simple déroulage de boucle ne résout pas le coût dominant observé ici. Une voie W4A16/Tensor Cores devra être évaluée séparément avec qualification numérique.


## Suite : regroupement de lignes et partage des activations

`ffn_group_probe` compare quatre variantes au même kernel de référence : 2 ou 4 groupes de quatre lignes par CTA, avec lecture directe des activations (g2/g4), ou copie des activations en mémoire partagée par CTA (s2/s4). Les partitions K et la réduction FP32 restent identiques. Les groupes hors limites du dernier CTA ne lisent aucun poids mais participent aux barrières.

Sur les matrices concaténées gate/up des blocs 0 et 5 :
- Référence chaude : environ 44–49 µs ; s4 : 36–38 µs.
- Référence après éviction : environ 120–124 µs ; s4 : 122–126 µs ; s2 : 127–128 µs.
- g4 reste proche de la référence après éviction (119–124 µs), sans séparation convaincante ; g2 également.

Packing inverse, égalité bit à bit et oracle CPU réussis pour les quatre variantes, les deux matrices réelles, trois entrées et les trois tailles synthétiques (y compris CTA incomplet). Formatage et Clippy des exemples réussis. Source de l'expérience : `kernels/ada/q4_gate_grouped.cu`, lanceur `crates/cuda-backend/examples/ffn_group_probe.rs`, résultats `benchmarks/ffn-group-probe.txt`.

Décision : aucune activation en production, aucun gain en tokens/s revendiqué. Pas de benchmark moteur pour ces variantes sans avantage net à froid. Cette expérience ne prouve pas la saturation DRAM : les compteurs matériels restent nécessaires pour attribuer précisément la limite. Elle montre que la réutilisation des activations au niveau CTA ne suffit pas dans ce test de flux de poids.
