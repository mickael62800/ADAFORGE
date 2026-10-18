# Architecture

## Dépendances

Le domaine ne dépend d’aucune autre crate ni bibliothèque externe.
L’application dépend du domaine et définit les ports dont elle a besoin.
Les adaptateurs dépendent de l’application et/ou du domaine, jamais l’inverse.
La crate `app` assemble les implémentations concrètes.

Le premier port sortant est `GpuDiagnostics`, défini dans application. Le cas d’usage `check_gpu` l’appelle ; le backend CUDA l’implémente et la CLI présente son rapport. Les types CUDA ne traversent pas ce port.

## Frontière GPU

Le backend possède les allocations CUDA, streams, événements, cache KV et graphes CUDA. Les types CUDA ne traversent pas ses interfaces publiques avec l’application.
Les abstractions portent sur des opérations complètes. La boucle de calcul reste dans le backend pour éviter copies, allocations et synchronisations imposées par les couches.
Le chargement de poids devra permettre un accès efficace aux données sans imposer une copie intégrale supplémentaire en RAM.

## Rust et CUDA

Rust assure l’orchestration. Nos kernels sont écrits en CUDA C++ et ciblent sm_89. Aucun moteur d’inférence existant n’est intégré.
Les appels bas niveau et blocs `unsafe` seront confinés, documentés et vérifiés dans le backend. Le kernel de diagnostic est compilé à l’exécution avec NVRTC pour compute_89. Les buffers sont détenus par les objets RAII de cudarc ; les deux lancements unsafe sont documentés dans le backend.

## Validation

La justesse numérique, la mémoire et la performance GPU seront vérifiées séparément. Les benchmarks compareront des modèles, quantifications et conditions équivalents. Aucun gain de performance n’est acquis avant mesure.

## Inspection des modèles

Le domaine définit TensorInfo et WeightType. L’application expose le port ModelInspector et le cas d’usage inspect_model. GgufInspector implémente le port dans model-formats. La CLI présente le rapport sans connaître le lecteur concret ; app assemble les adaptateurs. Aucun type CUDA ou lecteur de fichier ne traverse le domaine.

## Diagnostic matrice-vecteur

`Q4Matrix` dans le domaine possède un stockage validé, expose ses dimensions en lecture seule et fournit un oracle CPU FP64. Le port `ModelInspector` permet de lire une matrice Q4_0 ; l’adaptateur GGUF ouvre le fichier, valide son inventaire et lit la seule plage demandée. Le cas d’usage `check_model_matvec` relie ce port au diagnostic GPU sans dépendre de CUDA. La CLI n’importe aucun adaptateur concret.

Le backend reçoit les blocs compressés, alloue ses buffers RAII, compile notre kernel et compare les résultats. Le diagnostic n’est pas encore l’interface d’exécution du futur moteur : le chargement permanent, la réutilisation des buffers et l’enchaînement des couches seront ajoutés séparément.

## RMSNorm

Le domaine possède les poids et epsilon validés dans `RmsNorm`, ainsi que l’oracle CPU FP64. Le lecteur GGUF récupère le vecteur F32 et epsilon ; le cas d’usage `check_model_rms_norm` invoque le port diagnostic. Le backend possède les buffers, la réduction CUDA et la comparaison numérique. Les appels CUDA restent dans cet adaptateur.

Les diagnostics RMSNorm et matvec restent accessibles séparément. Le diagnostic projection les enchaîne désormais sur des buffers GPU résidents.

## Projection résidente

Le cas d’usage `check_model_projection` lit les deux tenseurs du premier bloc et vérifie la compatibilité des dimensions. Le backend `projection` compile les kernels existants, conserve les poids et buffers pour toutes les entrées et répétitions du diagnostic, puis lance les deux opérations sur son stream. Seule la sortie finale est copiée vers le CPU pour validation. Ces ressources sont libérées par RAII à la fin du diagnostic ; la résidence permanente d’un modèle complet n’est pas encore implémentée.

## Q/K/V

Le backend de projection utilise maintenant une chaîne commune pour une ou plusieurs matrices partageant la même entrée normalisée. Les anciens diagnostics restent disponibles. L’application choisit les tenseurs du bloc demandé et traite explicitement l’absence de poids V pour Gemma 4. Les poids sont conservés pendant le diagnostic ; seuls les résultats finaux sont rapatriés pour contrôle.

## Normalisation des têtes

Le chargement des poids QKV est commun aux diagnostics bruts et normalisés. Le backend ajoute un mode à la chaîne existante avec deux normalisations par tête. Les sorties brutes et normalisées sont distinctes pour préserver V lorsque K/V partagent les poids. Le nombre de têtes est déduit des dimensions vérifiées ; le même kernel RMSNorm est réutilisé avec un bloc par tête. Aucune donnée de projection GPU n’est rapatriée pour construire la référence CPU.

## RoPE

Le domaine `Rope` valide largeur paire, base, facteurs positifs et limite de contexte. Il fournit une référence CPU et les fréquences inverses. L’application sélectionne les paramètres locaux ou globaux à partir des métadonnées et charge les facteurs via le port de lecture F32. Le backend ajoute deux lancements de notre kernel NeoX à la chaîne existante ; des buffers distincts préservent les sorties non tournées, notamment V. Les positions sont des entrées de diagnostic : il n’y a pas encore d’état de séquence ni de cache KV.

## Préparation V

Le diagnostic `vnorm-check` étend la chaîne positionnelle existante avec un buffer de sortie V distinct. La réduction RMSNorm CUDA est commune, avec spécialisation compile-time pour supprimer la lecture et la multiplication d’échelle sur V. Le même epsilon du modèle est utilisé. Les cas d’usage RoPE et V partagent le chargement des paramètres ; leurs modes restent explicitement distincts.

## Cache KV

Le domaine `KvLayout` vérifie les dimensions et les tailles sans dépendance CUDA ; `KvState` protège la longueur et les plages d’ajout. Le backend `KvCache` possède deux buffers RAII sur un stream, ajoute des données résidentes par copies device-to-device et ne publie la nouvelle longueur qu’après mise en file des deux copies. Une erreur partielle invalide l’objet. Le port de diagnostic ne révèle que le rapport de validation.

Le cache initial concerne une couche/séquence et ne contient aucune politique de fenêtre glissante ni gestion multi-requêtes. Le diagnostic emploie la géométrie du modèle mais des données synthétiques ; le diagnostic isolé reste disponible en complément du branchement décrit ci-dessous.

## Intégration chaîne/cache

Le mode `kv-chain-check` de la chaîne positionnelle possède un `KvCache` sur le même stream. Il ajoute les buffers K après RoPE et V après norme directement, une seule fois par position. Le rapport contient les contrôles du préfixe cumulé et les métriques de comparaison CPU. Les diagnostics antérieurs restent inchangés et ne remplissent pas le cache. Il s’agit d’une séquence de huit activations synthétiques avec poids réels, pas encore du runtime de génération.

## Attention

Le domaine `AttentionConfig` protège la géométrie GQA, la fenêtre et l’échelle, et fournit un oracle CPU stable en FP64. Le backend `AttentionKernel` possède son module compilé, son buffer de sortie et son stream. Le cache expose une vue résidente typée (K, V, longueur logique, géométrie) interne au backend. Le kernel ne modifie ni Q ni le cache.

L’application lit la fenêtre locale depuis les métadonnées et sélectionne l’échelle Gemma 4 à 1. Le diagnostic d’attention étend la chaîne avec cache, sans transfert hôte pour alimenter le calcul. Les sorties GPU sont relues seulement après l’attention pour comparaison. Le diagnostic isolé couvre la causalité avec tokens futurs, ce que la séquence d’intégration 0–7 ne peut pas démontrer seule.

## Sortie d’attention et résiduel

Le domaine `AttentionOutput` possède une matrice Q4_0 et une norme post-attention de tailles compatibles, et fournit la référence CPU de la branche finale. Le backend `OutputKernel` réutilise nos kernels matvec, RMSNorm et addition, avec trois buffers de sortie distincts. Il consomme directement le buffer de l’attention et l’entrée brute résidente.

Les options finales du diagnostic sont regroupées dans `ChainTail` pour éviter d’ajouter un paramètre à chaque nouveau calcul. Le choix côté application utilise `AttentionStage` (sans attention, attention brute, branche de sortie) ; les diagnostics existants conservent leur périmètre.

## FFN et bloc dense

Le domaine `Ffn` valide les dimensions des trois matrices et des deux normes, conserve l’échelle finale et fournit l’oracle CPU. Le backend `FfnKernel` détient les poids GPU et sept buffers réutilisés ; il réutilise les kernels Q4_0/RMSNorm et ajoute l’activation GELU gate/up et l’ajout résiduel avec échelle finale.

`LayerWeights` regroupe les références nécessaires à un bloc pour éviter une interface à trop de paramètres. `ChainTail` contient maintenant un FFN optionnel ; les modes précédents n’en ont pas. Le port diagnostic couche appelle la même chaîne d’attention puis le FFN sur le même stream. Le domaine ne connaît ni CUDA ni le format GGUF.

## Embeddings

Le domaine `EmbeddingBatch` contient les IDs, dimensions et lignes Q6_K sélectionnées et validées ; il fournit le décodage CPU et l’échelle d’entrée. Le port de lecture GGUF accepte des IDs numériques et lit les plages correspondantes sans charger la table complète. Le port GPU compile notre déquantification, compare les versions brutes et mises à l’échelle, puis rend un rapport sans type CUDA.

Cette brique est actuellement testée séparément des blocs Transformer. Les poids compactés correspondent aux seules lignes demandées ; la résidence complète et le lookup GPU avec IDs restent à décider au moment de l’assemblage du runtime.

## Entrée résidente de la première couche

`GpuEmbeddings` conserve la fonction CUDA, les lignes compressées sélectionnées et leur sortie déquantifiée. Il expose une copie de ligne GPU/GPU bornée vers le buffer d’entrée. `ChainTail` peut maintenant fournir un `EmbeddingBatch` : l’oracle CPU utilise son décodage indépendant et le chemin GPU utilise exclusivement le résultat du kernel d’embedding. Le mode synthétique précédent reste disponible.

L’application réserve ce mode à la couche 0 et huit IDs pour le diagnostic avec cache actuel. Le contrat bas niveau est séparé du futur tokenizer et du chargement d’une table entièrement résidente.
