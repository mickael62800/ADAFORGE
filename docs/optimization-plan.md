# Plan consolidé des optimisations AdaForge / RTX 4090

Objectifs : dépasser llama.cpp CUDA sur le même modèle, puis atteindre au moins **100 tokens/s de décodage pour une conversation**. Publier séparément le débit sans spéculation, le débit avec spéculation, le prefill et le temps du premier token. Aucun résultat Marlin ou Imp sur une autre carte ou un autre modèle ne constitue une mesure de notre moteur.

État de référence actualisé après déchargement du modèle utilisateur et conversion matérielle des échelles : **76.10 tokens/s pour AdaForge, 78.39 pour llama.cpp CUDA**, sur le même prompt court et sept répétitions. AdaForge reste derrière. Voir [mesures, portée et essais](optimization-hardware-half.md). Les séries antérieures ne doivent pas être additionnées ni assimilées à ce face-à-face.

P0 = mesurer avant de choisir ; P1 = essais immédiats sans changement nécessaire de précision ; P2 = kernels structurants et chemin par lots ; P3 = dépendances ou objectifs de contexte/démarrage ; P4 = autre objectif de service. Ce catalogue regroupe les pistes identifiées et leurs dépendances ; il ne prétend pas épuiser toutes les optimisations possibles.

| ID | Origine | Piste | État | Priorité | Condition / limite |
|---|---|---|---|---|---|
| M01 | Mesures | Élargir les mesures à plusieurs prompts, longueurs de sortie et contextes | À faire | P0 | Éviter de régler le moteur sur le seul Bonjour ; séparer chargement, TTFT, prefill et decode |
| M02 | Mesures | Profil récent du FP32 avec partage K down : temps GPU, soumission et attentes hôte | À faire | P0 | Le profil détaillé antérieur précède ce dernier réglage ; quantifier la marge CPU/GPU |
| M03 | Ada / Imp | Balayer la préférence L1/shared par famille de kernels | FFN 0/100 testé, aucun gain établi | P1 | Réglages retirés ; autres familles et nouveaux pipelines à qualifier ; voir optimization-carveout.md |
| M04 | Imp / AdaForge | Deux ou quatre lignes par warp, avec activations réutilisées | Deux lignes voisines testées, régression | P1 | Variante retirée ; voir optimization-gate-reuse.md ; autres mappings à justifier avant essai |
| M05 | Imp / Marlin | Activations en shared ou registres ; disposition sans conflits de banques | Vecteur complet en shared testé, régression | P1 | Variante retirée ; un pipeline par tuiles reste distinct de cet essai |
| M06 | Marlin / AdaForge | Rangement des poids Q4 et échelles par tuiles, groupe 32 conservé | Conservé pour FFN 3840/15360 | Fait ciblé | ABBA : +2,58 tok/s sur le prompt mesuré ; sans duplication GPU ; voir optimization-async-and-abba.md |
| M07 | Marlin | Lectures vectorisées alignées et moins de calculs d’adresses | À tester | P1 | Couplé à M06 ; pas de lecture uint4 aveugle sur chaque bloc de 18 octets |
| M08 | Marlin | Politiques de cache distinctes pour poids en flux et activations réutilisées | À tester | P2 | Les intrinsics cache testés précédemment ne couvraient pas ce pipeline complet |
| M09 | Marlin / Ada | Chargements asynchrones et pipeline à 2/3/4 étapes | Deux buffers, 1/4 groupes testés : plus lents | P2 | Variantes de recherche écartées ; un autre pipeline exige une hypothèse distincte |
| M10 | Marlin / AdaForge | Tuiles et partition des SM adaptées à chaque forme de matrice | À tester | P2 | Sélection hors chemin critique ; vérifier réductions inter-blocs et petites matrices |
| M11 | Marlin | Déquantification half2/LOP3 et calcul W4A16 sur Tensor Cores | À prototyper | P2 | Nouvelle précision, scales groupe 32, sorties et réductions à définir ; tester M=1 séparément |
| M12 | AdaForge | Optimiser la projection Q6_K de la tête | À tester | P2 | Poids, échelles, lectures et distribution du travail ; indépendant du kernel INT4 |
| M13 | AdaForge | Fusion tête Q6_K et réduction greedy | À prototyper | P2 | Conserver les égalités, le rejet des non-finis et les 262144 candidats |
| M14 | Imp / AdaForge | Inclure la tête et la sélection dans un graphe CUDA réutilisable | À tester | P2 | D’abord mesurer la part des quatre lancements et de la lecture hôte |
| M15 | Imp | Courtes rafales ou boucle de génération GPU autonome | À étudier | P3 | Compatibilité pilote/cudarc et gestion EOS, budget, cache et streaming |
| M16 | AdaForge / Marlin | Prefill réellement par lots, avec GEMM et attention causale par blocs | À construire | P2 | Accélère le prompt ; brique nécessaire à la vérification spéculative |
| M17 | Imp | Vérification parallèle de propositions et annulation du KV rejeté | À construire | P3 | Dépend de M16 ; préserver les positions et l’historique exacts |
| M18 | Imp | Propositions suffix/ngram sans modèle supplémentaire | À tester après M17 | P3 | Gain dépendant des répétitions ; comparer sur texte réel |
| M19 | Google / Imp | Drafter MTP officiel compatible avec notre Gemma QAT | À tester après M17 | P3 | Chargement, activations partagées, coût de vérification et taux d’acceptation |
| M20 | AdaForge | Cache KV FP16, puis éventuellement autres précisions | À évaluer | P3 | Nouvelle précision ; surtout utile quand le contexte augmente |
| M21 | AdaForge / Imp | Cache local circulaire respectant la fenêtre 1024 | À construire pour contexte long | P3 | Réduit la croissance mémoire locale ; conserver les positions RoPE absolues |
| M22 | Imp / AdaForge | Attention groupée réutilisant K/V entre têtes Q et découpage pour contexte long | À tester | P3 | La distribution optimale dépend du contexte et des couches locales/globales |
| M23 | AdaForge | Réduction des buffers, copies internes et petites opérations par fusion ciblée | À profiler | P2 | Conserver les frontières d’arrondi ou documenter leur changement ; coût maximal à estimer |
| M24 | AdaForge | Cache de compilation PTX/modules et amélioration du chargement | À faire séparément | P3 | Améliore le démarrage, pas les tokens/s après échauffement |
| M25 | AdaForge | Traitement de plusieurs conversations et cache de préfixes | Hors priorité actuelle | P4 | Objectif distinct du débit d’une conversation ; une seule carte reste suffisante comme cible |
| M26 | NInfer-4090 / ExLlamaV2 | Qualifier hors génération un schedule Q4 par forme Gemma, puis dispatcher par table fixe | À prototyper | P1 | Essais limités aux formes réelles ; conserver un choix déterministe si les mesures sont proches ou instables |
| M27 | NInfer-4090 / Redline | Tuiles Q4_0 de codes et paires d’échelles alignées, `cp.async` et double-buffering, avec accumulation FP32 | Tuiles conservées ; async 1/4 groupes écarté | P2 | Les copies asynchrones ne gagnent pas sur ce layout ; pas d'intégration de ces variantes |
| M28 | NInfer-4090 / Redline | Étendre l’attention GQA à un chemin contexte long : CTA par tête KV, softmax en ligne et accumulations en registres | À étudier après profil long contexte | P3 | Utile surtout quand les lectures KV dominent ; vérifier couches locales et globales Gemma séparément |
| M29 | NInfer-4090 | KV INT8 puis 4-bit pour les très longs contextes | À évaluer après M28 | P3 | Changement de précision : oracle propre, qualité et rappel long contexte requis ; pas une optimisation prioritaire du prompt court |
| M30 | Redline | Contrat de benchmark reproductible : entrées, paramètres, résultats bruts, dispersion et profil Nsight node-level | À faire avec M01–M02 | P0 | Ne modifie pas le moteur ; condition pour accepter un gain de performance |
| T01 | AdaForge | Poids, buffers et caches KV résidents ; graphes CUDA | Conservé | Fait | Pas de recalcul CPU des couches ni de rechargement entre tokens |
| T02 | AdaForge | Sélection greedy sur GPU et retour de 8 octets | Conservé | Fait | Ne plus compter un gain hypothétique de suppression du transfert complet des logits |
| T03 | AdaForge | QKV regroupé, K/V commun calculé une seule fois, gate/up regroupé | Conservé | Fait | Le partage des poids ne fusionne pas les caches K après RoPE et V |
| T04 | AdaForge | Réductions RMSNorm/attention, Q4 et Q6 optimisés | Conservé | Fait | Oracle CPU et tolérances initiales préservés |
| T05 | Imp / AdaForge | Partage K entre quatre warps sur down FFN | Conservé | Fait | Gain mesuré ; extension à toutes les projections non retenue |
| T06 | Imp / AdaForge | Activations Q8 et dp4a dans le FFN | Expérimental, désactivé par défaut | Recherche | Gain faible ; IDs changés ; qualité générale non évaluée |
| R01 | AdaForge | Séparation simple des échelles et octets Q4 par ligne | Écarté dans cette forme | Archivé | 66.24 tok/s contre environ 68.1 avant/après |
| R02 | AdaForge | Balayage de 1/2/8/16 warps en gardant une ligne par warp | Aucun gain net | Archivé | Ne réfute ni multi-lignes par warp ni pipeline par tuiles |
| R03 | AdaForge | Gate et up calculés conjointement par warp | Écarté dans cette forme | Archivé | Plus lent malgré la validation numérique |
| R04 | AdaForge | LOP3 FP32 seul et conversion matérielle du scale seule | Aucun gain net | Archivé | Ne réfute pas la déquantification half2 intégrée à un autre kernel |
| X01 | Imp / NVIDIA | PDL, NVFP4 natif sm_120a, TMA/WGMMA spécifiques aux générations suivantes | Non retenu pour sm_89 | Exclu | Vérifier chaque capacité ; ne pas transposer le chemin Blackwell |
| X02 | Imp | Caches d’experts MoE et offload de modèles trop gros | Sans objet pour notre Gemma dense résidente | Exclu | Ne contribue pas à notre comparaison actuelle |
| X03 | Suggestion initiale | Maintenir tout le KV local FP32 dans le L2 | Impossible intégralement aux dimensions visées | Exclu | 640 Mio pour 40 couches à fenêtre 1024 ; une politique de persistance ne verrouille pas le cache |

## Ordre de travail retenu pour les prochains essais

État le plus récent : [qualification ABBA et essais async](optimization-async-and-abba.md). Tuiles synchrones retenues par défaut pour FFN 3840/15360 (+3,7 % dans la série ABBA), sans poids dupliqués. Les variantes async sont plus lentes. Poursuivre sur les autres formes, la tête Q6 et les graphes, tout en élargissant les mesures et les profils decode-only.

Dernière passe : [essai L1/shared et profil actualisé](optimization-carveout.md). Les deux préférences FFN testées n'ont pas démontré de gain. La trace actuelle confirme que gate/up Q4 reste la première cible GPU ; elle mélange encore prefill et decode.

Passe suivante : [réutilisation des activations gate/up](optimization-gate-reuse.md). Deux lignes par warp et copie complète des activations en shared ont régressé ; les deux variantes ont été retirées. Poursuivre sur le rangement par tuiles et les lectures des poids en tenant compte de R01.

Qualification initiale : [Q4 par tuiles](optimization-tiled-q4.md). Packing réversible et réduction FP32 corrigée validés sur 48 blocs ; 72,55 et 72,76 tok/s sur deux prompts du prototype initial. Duplication GPU supprimée ; troisième prompt à 96 tokens : 65,53 contre 62,96/64,98. Le chemin était encore optionnel à ce stade. La qualification ABBA et les essais cp.async suivants sont décrits ci-dessus.

1. M01–M02/M30 : étendre le protocole ABBA aux contextes et profiler le decode seul des deux moteurs, sans mélanger débit agrégé et débit d’une conversation.
2. M10/M26 et M03 : qualifier les autres formes de projections, les partitions de K et préférences de cache encore non testées. Ne pas relancer les variantes FFN écartées sans hypothèse différente.
3. M12–M14 : mesurer et optimiser la tête Q6 et ses soumissions, puis l'inclure dans un graphe avec sa sélection.
4. M23 : fusionner les petites opérations réellement coûteuses d'après le profil, avec conservation des frontières FP32.
5. M11/M16 : chemin Tensor Cores et prefill par lots, avec précision explicite comparée à FP32.
6. M17–M19 : vérification de propositions, ngram et drafter ; mesurer le coût réel par token accepté.
7. M20–M22/M28–M29 : cache KV et attention à contexte long ; M24–M25 restent des objectifs distincts de démarrage et de service.

## Conditions d’adoption

Pour chaque essai : isoler le changement, un échauffement puis plusieurs mesures avant/après/avant, poids et contexte identiques, absence de concurrence GPU pendant le benchmark, conservation des résultats bruts. Étendre à plusieurs prompts et tailles de contexte avant de généraliser.

Pour un changement qui conserve les calculs FP32 : oracle CPU indépendant, tests des limites/positions/reset, contrôle complet des couches/logits pour un changement mathématique significatif, et génération de référence. Pour un changement de précision : conserver le chemin FP32, vérifier d’abord le calcul avec son propre oracle, mesurer séparément l’écart à FP32, puis évaluer logits/perplexité et réponses sur plusieurs textes avant toute activation par défaut. Une simple réponse cohérente ou le seul passage de tests de kernels ne valide pas la qualité générale.

Les gains hypothétiques ne s’additionnent pas. Les optimisations déjà écartées ne seront rouvertes que si le kernel, le rangement ou le contexte ont changé de manière à justifier une nouvelle hypothèse.

Références : [analyse Marlin](marlin-review.md), [analyse Imp](imp-review.md), [analyse des moteurs externes](external-engines-review.md), [prototype Q8](q8-experiment.md), [guide NVIDIA Ada](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html).


## Essai Q6 guidé par le profil — 18 octobre 2026

Le regroupement de huit lignes par bloc a été validé numériquement puis comparé en ABBA : -3,26 % sur la moyenne des médianes, IDs identiques. Il reste désactivé ; quatre lignes par bloc restent le défaut. Voir [protocole et résultats](optimization-q6-launch.md).


## Tête greedy sans softcap inutilisé — 18 octobre 2026

Spécialisation Q6 brute validée bit pour bit, génération identique. ABBA : +0,614 % avec dérive des contrôles -3,182 %, donc gain non établi et expérimentation désactivée. Voir [résultats](optimization-q6-raw.md).


## Fusion RMSNorm/RoPE — 18 octobre 2026

Fusion Q/K validée bit pour bit sur têtes 256/512 et grandes positions ; oracle CPU 48 blocs réussi (ratio maximal 0,42836). Trace : 96 lancements en moins par token, environ 0,107 ms/étape de moins sur la famille RMSNorm/RoPE. Deux ABBA à 24/64 tokens indiquent environ +1 %, avec variations de contrôle comparables ou supérieures : expérimentation désactivée par défaut. Voir [rapport et décision](optimization-norm-rope.md). Prochaine hypothèse : regrouper davantage la préparation Q/K/V.


## Préparation Q/K/V groupée — 18 octobre 2026

Adoptée par défaut pour têtes 256/512 après validation bit pour bit ciblée et oracle CPU 48 blocs. 192 lancements supprimés par token. ABBA court : environ +2,42 % (+1,80 token/s) ; test long positif mais contrôles plus variables. Retour séparé : `ADAFORGE_EXPERIMENTAL_QKV_PREP=0`. Voir [rapport complet](optimization-qkv-prep.md).


## Attention et écritures KV — 18 octobre 2026

Deux expériences supplémentaires : [écriture KV fusionnée](optimization-prep-cache.md), correcte mais +0,50 % avec dérive des contrôles -4,04 %, donc désactivée ; [exponentielles du softmax en parallèle](optimization-attention-exp.md), exacte sur les cas testés, +3 % sur génération longue mais petit recul descriptif au court, donc expérimentale. Les deux chemins restent FP32. La préparation Q/K/V groupée précédemment adoptée reste active. La télémétrie GPU avant/après les séries est désormais disponible via `--gpu-telemetry`.


## Conversion matérielle des échelles et reprise de comparaison

Conversion FP16→FP32 native adoptée après test exhaustif des 63488 valeurs finies et ABBA positif. Rangement Q6 par quatre lignes testé et laissé désactivé. Le face-à-face courant donne 76,10 contre 78,39 tokens/s. Priorité aux projections dominantes ; le seuil de 100 demande environ 3,14 ms/token de moins. [Rapport](optimization-hardware-half.md).


## Fusion gate/up et GELU

Prototype optionnel testé sur deux prompts en ABBA : +0,59 % court, environ -2,51 % sur 64 tokens. Désactivé par défaut ; oracle CPU couche réelle réussi et IDs identiques. Voir [rapport](optimization-fused-gelu.md).
