# Analyse de Marlin pour AdaForge / RTX 4090

Source récupérée dans `D:/ADAFORGE/tools/marlin-reference`, commit `1f25790bdd49fba53106164a24666dade68d7c90`. Dépôt : https://github.com/IST-DASLab/marlin. Licence Apache-2.0 conservée dans cette copie. Aucune installation de PyTorch, aucune compilation ou intégration de Marlin dans AdaForge : il s’agit ici d’une analyse des sources pour notre moteur indépendant. Aucun benchmark Marlin n’a été exécuté sur notre GGUF.

## Lecture du kernel

| Mécanisme observé | Emplacement | Conséquence pour AdaForge |
|---|---|---|
| INT4 déquantifié en fragments half2 par LOP3 puis opérations FP16 | `dequant`, `scale` dans le kernel CUDA | Essai distinct de notre LOP3 FP32 écarté. Le calcul des poids et produits FP16 peut ajouter des arrondis ; valider les logits et la qualité. |
| Tensor Cores `mma.sync` avec entrées FP16 et accumulateurs FP32 | `mma` | Candidat pour un chemin W4A16 propre, surtout pour prefill et vérification de plusieurs tokens ; sortie et réductions de Marlin ne restent pas toutes FP32. |
| Chargements global→shared de 16 octets avec `cp.async` | `cp_async4_pred`, `cp_async4_stream` | Le rangement doit garantir l’alignement. Les blocs natifs GGUF de 18 octets ne permettent pas de remplacer chaque lecture par un uint4 directement. |
| Politique d’éviction prioritaire pour les poids consommés une seule fois | `cp_async4_stream` | Tester séparément poids en flux et activations réutilisées, plutôt que demander la même politique de cache à tous les buffers. |
| Pipeline global/shared/registers et quatre étapes de chargement | `start_pipes`, boucle pipeline, `STAGES` | Masquer la latence par anticipation et double buffering ; mesurer 2/3/4 étapes selon la mémoire partagée et les registres de la 4090. |
| Rangement préalable des poids et permutation des échelles | `_get_perms`, `Layer.pack` | Le rangement et le kernel sont conçus ensemble. Notre premier rangement Q4 écarté ne réfute pas ce rangement par tuiles. |
| Fragments Tensor Cores chargés avec `ldmatrix`, accès shared réorganisés | `ldsm4`, offsets shared | Adapter le placement pour éviter les conflits de banques et les transformations répétées. |
| Partition des tuiles en bandes entre SM | `Marlin`, `slice_*`, `iters` | Meilleure répartition sur les formes réelles ; étudier le coût des réductions et verrous entre blocs. |
| Paramètres de tuiles et boucles spécialisés à la compilation | templates, `CALL_IF`, `thread_k`, `thread_n` | Plusieurs kernels pour nos dimensions fixes, sélectionnés après mesures et sans recherche coûteuse pendant la génération. |
| Réductions locales puis, si nécessaire, réductions globales via la sortie | `thread_block_reduce`, `global_reduce`, `write_result` | Mesurer le compromis entre occupation, trafic et arrondis. Marlin utilise aussi du stockage FP16 dans certaines réductions : ce n’est pas une équivalence automatique au chemin FP32 actuel. |

Sources principales : [kernel CUDA](https://github.com/IST-DASLab/marlin/blob/1f25790bdd49fba53106164a24666dade68d7c90/marlin/marlin_cuda_kernel.cu), [rangement et interface](https://github.com/IST-DASLab/marlin/blob/1f25790bdd49fba53106164a24666dade68d7c90/marlin/__init__.py), [présentation et protocole](https://github.com/IST-DASLab/marlin/blob/1f25790bdd49fba53106164a24666dade68d7c90/README.md).

## Incompatibilités à résoudre avant un prototype

- `Layer` accepte des groupes de 128 ou une échelle par colonne ; notre Q4_0 porte une échelle tous les 32 poids. Le dispatch CUDA de cette version instancie ces configurations, pas notre groupe 32. Il faut créer une stratégie de tuiles/scales adaptée, pas fusionner arbitrairement quatre échelles en une.
- `Layer.pack` part de poids FP16 et les arrondit sur la grille. Un rangement sans perte de notre GGUF devra préserver directement les nibbles et les bits des échelles, puis permettre de reconstruire le stockage source exactement.
- Le wrapper impose une entrée multiple de 128 et une sortie multiple de 256. Nos principales dimensions de projections (3840, 15360, 4096, 2048, 8192 et 512) respectent ces divisibilités. C’est favorable, mais les contraintes du dispatch et des groupes de scales doivent encore être vérifiées pour chaque configuration ; aucun padding ne doit être ajouté sans nécessité mesurée.
- L’interface prend et renvoie du FP16. Notre chemin validé conserve FP32. Les choix de conversion, de sortie, de réductions et de résiduels doivent être explicites et testés.
- La tête du modèle est Q6_K. Un kernel INT4 ne remplace pas directement cette tête ; elle demande un chantier séparé.
- Un bon résultat de GEMM à plusieurs tokens ne prouve pas un gain sur notre décodage à un token. Mesurer séparément M=1 et plusieurs valeurs de M pour le prefill et la vérification spéculative.
- Les constantes originales (256 threads, quatre étapes, 96 Kio shared) sont des configurations à tester, pas les paramètres automatiquement optimaux de notre 4090.

## Expériences à dériver

1. Rangement Q4 par tuiles, réversible bit à bit, avec groupe 32 et alignement 16 octets.
2. Prototype W4A16 isolé comparé au chemin FP32 et au prototype Q8, sur nos matrices et des activations représentatives.
3. Mesures des politiques de cache, du nombre d’étapes, des tuiles et de la pression sur les registres.
4. Chemin par lots et réutilisation des poids pour prefill/vérification, avant l’intégration d’un drafter.

La décision d’adoption se fait sur le temps complet et la précision mesurée. Aucun facteur de vitesse publié par Marlin n’est transposé à AdaForge.
