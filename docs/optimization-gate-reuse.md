# Réutilisation des activations gate/up : deux essais

Ces essais ciblent le kernel Q4_0 `q4_gate_up` du FFN. Les poids restent au format GGUF natif, les activations FP32 et les quatre accumulateurs par ligne sont conservés. Les changements ne concernent pas le chemin expérimental Q8.

| Variante | Avant | Variante | Avant répété | Décision |
|---|---:|---:|---:|---|
| Deux lignes voisines de la même matrice par warp ; quatre lectures d'activations partagées entre les lignes | 69,28 | 64,03 | 73,26 | Régression : retirée |
| Une ligne par warp ; vecteur d'entrée copié en shared une fois par CTA de quatre warps | 71,64 | 63,59 | 70,64 | Régression : retirée |

Débits médians en tokens/s, cinq répétitions après un échauffement par processus. Protocole : même modèle, prompt Bonjour, 24 tokens de sortie, reset KV et séquence avant/après/avant ; les IDs sont identiques pour toutes les répétitions. Données brutes : `benchmarks/optimization-gate-two-rows.json` et `benchmarks/optimization-gate-shared-x.json`. La première série montre une dérive de la référence, mais la variante reste inférieure aux deux contrôles. Les taux ne sont pas généralisés à d'autres prompts ou contextes.

La variante à deux lignes possède davantage d'accumulateurs vivants ; celle en shared ajoute une copie et une synchronisation par CTA. Ce sont des explications possibles, pas un diagnostic mesuré des stalls ou de l'occupation. Ces essais montrent seulement que ces deux organisations ne gagnent pas dans les conditions testées.

Validation : les 26 tests GPU existants passent sur la variante deux lignes. Un test supplémentaire compare séparément gate et up à l'oracle CPU, avec 1/5/8/9 lignes, 32/96/384 colonnes, échelles nulles, négatives et subnormales. Il passe sur la variante deux lignes puis sur shared ; les 27 tests GPU passent sur shared. Les tolérances initiales sont conservées.

Le moteur de production a retrouvé ses kernels et sa géométrie précédents. Le test de couverture des dimensions impaires est conservé avec la géométrie de production. Les sources et binaires des variantes sont archivés dans `tools/optimization-gate-two-rows/` afin de ne pas répéter les mêmes essais.

Conséquence pour la suite : M04–M05 sont partiellement explorées ; on ne doit pas retester ces variantes sans modifier une autre hypothèse. La prochaine expérience structurante concerne le rangement des poids par tuiles et les lectures alignées (M06–M07/M27), avec vérification réversible du Q4_0 groupe 32. Une simple séparation des scales par ligne a déjà été écartée (R01) ; un nouveau prototype doit changer la distribution des poids entre threads et le pipeline, et pas seulement les pointeurs.
