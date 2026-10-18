# Reprise des mesures et déquantification matérielle — 18 octobre 2026

L'utilisateur a signalé puis déchargé un autre modèle. Les anciennes séries ne sont pas déclarées invalides sans preuve, mais leur attribution causale est limitée : la concurrence GPU n'était pas documentée de manière continue. Au début de cette reprise, nvidia-smi indiquait 2921 Mio utilisés, 8 % d'activité et 35 °C. Le processus llama-server utilisateur encore ouvert est un service d'embeddings bge-m3 avec `-ngl 0`, laissé intact. Cela ne constitue pas une preuve d'exclusivité matérielle permanente.

## Modification retenue

`decode_half` (Q4) et `embedding_half` (Q6) utilisent l'instruction PTX `cvt.f32.f16` au lieu d'une reconstruction logicielle avec extraction d'exposant/mantisse et traitement du signe. Ce sont les échelles des poids déjà stockées en FP16 qui sont converties, pas les activations : les calculs restent FP32, les poids Q4_0/Q6_K et leurs valeurs sont inchangés.

Test exhaustif : 63488 représentations FP16 finies, avec oracle arithmétique CPU FP64, résultats identiques bit pour bit dans les deux fonctions. Les échelles non finies sont refusées par les constructeurs du domaine. Les 32 tests GPU passent, ainsi que clippy sans avertissement et le formatage. Les identifiants de génération sont identiques dans les séries de performance.

ABBA de la conversion matérielle : contrôles [73.81980585391062, 71.80333129994625], variante [75.34017727871282, 75.65991067525155] tokens/s. Moyenne des médianes : 72.812 → 75.500, delta descriptif 3.692 %. Dérive des contrôles -2.732 %. Les deux médianes candidates dépassent les deux contrôles ; le pourcentage reste descriptif, pas une garantie générale. Modification retenue. Données : `benchmarks/optimization-hardware-half-abba.json`.

## Rangement Q6 expérimental

Une table Q6_K intercalée par quatre lignes est implémentée dans `kernels/ada/q6_tiled.cu`, activable par `ADAFORGE_EXPERIMENTAL_Q6_TILES=1`. Les payloads sont intercalés par octet et les échelles restent alignées sur 16 bits. Le kernel conserve l'ordre des sommes FP32 ; les deux premières étapes de réduction passent par la mémoire partagée. La table sert aussi aux lookups d'embeddings, statiques et dynamiques. Elle remplace la table native en VRAM : aucune duplication GPU. Une dernière tuile partielle est complétée à zéro ; le vocabulaire réel est divisible par quatre, donc pas de surcoût de poids GPU pour ce modèle. Le packing CPU temporaire et son temps s'ajoutent au chargement, hors débit de decode.

Tests : logits bruts identiques bit pour bit en largeurs 256/768/3840, neuf lignes avec tuile partielle, échelles signées/nulle/sous-normale ; tête face au CPU et embeddings en largeur 512 avec cinq lignes ; graphe/direct avec reset et tokens variables. Les IDs du benchmark réel sont identiques. Ce n'est pas une nouvelle validation CPU exhaustive des 48 couches du chemin tuilé.

ABBA Q6 tuilé : contrôles [73.07816345043315, 77.33106943825705], variante [74.3633366764503, 71.18699989198146] tokens/s. Delta descriptif -3.230 %, dérive des contrôles 5.820 %. Aucun gain démontré : variante désactivée par défaut. Données : `benchmarks/optimization-q6-tiles-abba.json`.

## Comparaison actuelle avec llama.cpp CUDA

Sept répétitions après un échauffement, prompt « Bonjour, comment vas-tu ? », 20 tokens de prompt et 24 de sortie, IDs identiques. AdaForge utilise le chemin de production avec conversion matérielle ; Q6 tuilé et les autres variantes expérimentales sont désactivés. Référence CUDA b11669, offload 49/49 vérifié, contexte 512, KV FP16 et Flash Attention auto. Les détails et limites des horloges de mesure sont dans le rapport : AdaForge temps backend, llama serveur via HTTP/SSE local. Chargement exclu ; les min/max suivants décrivent cette série et ne sont pas des intervalles de confiance.

| Moteur | Médiane tokens/s | Min–max |
|---|---:|---:|
| AdaForge | 76.103 | 72.963–77.172 |
| llama.cpp CUDA | 78.390 | 76.556–80.433 |

AdaForge reste derrière de 2.286 tokens/s sur ce test. Rapport conservé : `benchmarks/comparison-cuda-native-after-hardware-half.json`.

À 76.103 tokens/s, le temps par token est environ 13.140 ms. Atteindre 100 exige de retirer environ 3.140 ms/token : les seuls gains de quelques lancements ne suffisent pas. Dépasser la médiane actuelle de llama.cpp exige environ 0.383 ms/token de moins. Les prochaines priorités sont les projections Q4/Q6 et une comparaison de leur coût avec les kernels de référence ; toute variante de précision doit être évaluée explicitement. Les 100 tokens/s restent un objectif non démontré.
