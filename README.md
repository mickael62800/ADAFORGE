# AdaForge

Moteur d’inférence IA écrit en Rust avec des kernels CUDA C++ dédiés à la NVIDIA RTX 4090 (Ada, sm_89).

## État actuel

Premier diagnostic GPU opérationnel : détection RTX 4090, buffers GPU, compilation et exécution de notre kernel CUDA, validation CPU et chronométrage par événements CUDA. Aucun chargement de modèle, serveur HTTP ou génération de texte pour le moment.

## Démarrer

Installer une version stable récente de Rust prenant en charge l’édition 2024, puis exécuter depuis ce dossier :

```sh
cargo check --workspace
cargo run -p adaforge -- gpu-check
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
```

## Organisation

- `crates/domain` : types et invariants purs.
- `crates/application` : cas d’usage et ports.
- `crates/model-formats` : adaptateurs de lecture des modèles.
- `crates/cuda-backend` : exécution et mémoire GPU.
- `crates/server` : adaptateur HTTP et streaming.
- `crates/cli` : adaptateur de ligne de commande.
- `crates/app` : assemblage et exécutable `adaforge`.
- `kernels/ada` : futurs kernels CUDA C++ sm_89.
- `docs` : architecture et étapes de développement.
- `benchmarks` : protocole et résultats de mesure.
- `models` : fichiers locaux exclus du suivi Git.

Voir `docs/architecture.md` et `docs/roadmap.md`.

## Diagnostic GPU

```sh
cargo run -p adaforge -- gpu-check
cargo test --workspace
cargo test -p adaforge-cuda-backend real_gpu_check -- --ignored
```

Le diagnostic utilise le GPU d’indice 0 et exige une RTX 4090 de capacité 8.9. Il additionne 1 000 003 paires FP32 (environ 12 Mo de buffers GPU), vérifie tous les résultats et mesure une exécution après échauffement. Les transferts et la compilation sont exclus de ce temps. Les ressources sont libérées automatiquement en sortie.

Le backend utilise `cudarc 0.19.10`, avec les API CUDA 13.3 et chargement dynamique. L’exécution nécessite le pilote NVIDIA compatible et les bibliothèques NVRTC du CUDA Toolkit 13.3 accessibles via le PATH. La compilation Rust ne nécessite pas `nvcc` ni de compilation C++ par MSVC : NVRTC compile le kernel embarqué à l’exécution pour `compute_89`, puis le pilote le charge sur sm_89.

Le binaire peut être lancé depuis un autre dossier : le source CUDA est embarqué lors de la compilation Rust. Relancer `cargo build` après toute modification du kernel.

Les erreurs de calcul ou CUDA retournent un code de sortie non nul. Une bibliothèque CUDA/NVRTC absente peut provoquer un panic de chargement dans la dépendance cudarc ; vérifier le pilote et le PATH du Toolkit dans ce cas.

Ce diagnostic n’est ni un benchmark d’inférence ni une preuve d’accélération face à un autre moteur. Le temps varie avec l’activité du bureau et des autres applications GPU.

## Inspection du modèle

```powershell
cargo run -p adaforge -- model-info models/gemma-4-12b-it-qat-q4_0/gemma-4-12B-it-QAT-Q4_0.gguf
```

Le lecteur maison inspecte GGUF v3 little-endian sans charger les poids ni initialiser CUDA. Il conserve les métadonnées scalaires et les petits tableaux ; les tableaux dépassant 4096 éléments (notamment le vocabulaire) sont résumés, pas chargés pour une future tokenisation.

Il vérifie dimensions, tailles de blocs, alignements, débordements, doublons de noms, plages hors fichier et chevauchements. Les types de poids reconnus sont F32, F16, BF16, Q4_0, Q8_0 et Q6_K ; les autres sont refusés explicitement. Les limites de lecture sont volontaires : en-tête 256 Mio, chaîne 16 Mio, 100 000 tenseurs/métadonnées, 4 millions d’éléments par tableau, rang 1 à 4. Les tableaux imbriqués sont refusés.

Sur notre fichier : 667 tenseurs (338 F32, 328 Q4_0 et 1 Q6_K). Le nom du fichier ne suffit donc pas à identifier tous les formats internes. Cette validation porte sur la structure et les plages : elle ne garantit pas encore la compatibilité de l’architecture avec l’inférence et ne vérifie pas le contenu numérique des poids.

Références de format : https://github.com/ggml-org/ggml/blob/master/docs/gguf.md et https://github.com/ggml-org/ggml/blob/master/src/ggml-common.h . Aucun code d’un moteur existant n’est intégré.

## Premier calcul sur les poids de Gemma

```powershell
cargo run -p adaforge -- matvec-check models/gemma-4-12b-it-qat-q4_0/gemma-4-12B-it-QAT-Q4_0.gguf
cargo run -p adaforge -- matvec-check models/gemma-4-12b-it-qat-q4_0/gemma-4-12B-it-QAT-Q4_0.gguf blk.0.ffn_down.weight
cargo test -p adaforge-cuda-backend -- --ignored
```

`matvec-check` prend un chemin et un nom optionnel de tenseur (par défaut `blk.0.attn_q.weight`). Il lit uniquement la matrice sélectionnée, exige Q4_0 et un rang 2, et limite les données à 256 Mio. Les échelles FP16 non finies sont refusées. Les dimensions GGUF sont stockées dans l’ordre [colonnes, lignes].

Le GPU conserve les blocs Q4_0 compressés. Un warp calcule une ligne, avec quatre warps par bloc CUDA et une réduction par shuffle. Les poids sont déquantifiés à la volée ; aucune matrice FP32 intermédiaire n’est allouée. Il s’agit d’une première implémentation de justesse, pas d’un kernel déjà optimisé pour dépasser llama.cpp.

La vérification utilise trois vecteurs : zéro, motif déterministe et pseudo-aléatoire déterministe. Toutes les lignes sont comparées à un calcul CPU avec accumulation FP64. Une ligne échoue si `abs(GPU - CPU) > 1e-4 + 1e-4 * abs(CPU)` ; les valeurs non finies échouent aussi. Le diagnostic affiche le ratio maximal à cette tolérance (doit rester inférieur ou égal à 1).

Le temps affiché est la médiane de dix lancements sur le troisième vecteur après un échauffement. Lecture, compilation, transferts et référence CPU sont exclus du temps GPU. Cette mesure avec des poids réutilisés ne caractérise pas encore le débit d’un modèle complet.

Référence de représentation des blocs Q4_0 : https://github.com/ggml-org/ggml/blob/master/src/ggml-quants.c . Implémentations Rust et CUDA propres au projet.

## Normalisation RMSNorm

```powershell
cargo run -p adaforge -- rmsnorm-check models/gemma-4-12b-it-qat-q4_0/gemma-4-12B-it-QAT-Q4_0.gguf
cargo run -p adaforge -- rmsnorm-check models/gemma-4-12b-it-qat-q4_0/gemma-4-12B-it-QAT-Q4_0.gguf blk.0.attn_q_norm.weight
```

Le diagnostic lit un vecteur F32 du modèle Gemma 4 (par défaut `blk.0.attn_norm.weight`) et l’epsilon des métadonnées. Il applique `y[i] = x[i] / sqrt(mean(x²) + epsilon) * weight[i]`, sans ajouter 1 aux poids. Les échelles de Gemma 4 sont directement multiplicatives, contrairement à certaines anciennes variantes de Gemma.

Un bloc de 256 threads traite chaque vecteur. Le kernel est vérifié sur huit entrées : zéro, constante, très petite amplitude, alternance de signes, motif, pseudo-aléatoire déterministe, impulsion et forte amplitude. Toutes les sorties sont comparées à une référence FP64 avec `atol = rtol = 2e-5`. Les poids non finis et epsilon non positif sont refusés. Le diagnostic contrôle un opérateur isolé, pas encore une couche complète.

Le temps affiché concerne le lot entier de huit vecteurs : médiane de dix exécutions après échauffement, hors compilation, transferts et référence CPU. Les tests GPU synthétiques couvrent les largeurs 1, 31, 256, 257 et 3840. La réduction FP32 est prévue pour les amplitudes usuelles des activations ; des amplitudes proches des limites FP32 peuvent faire déborder la somme des carrés et ne sont pas prises en charge par ce premier kernel.

Références sémantiques : https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modeling_gemma4.py (`Gemma4RMSNorm`) et https://github.com/ggml-org/llama.cpp/blob/master/conversion/gemma.py (`Gemma4Model.norm_shift`). Le kernel est notre propre implémentation.

## Chaîne RMSNorm → projection Q

```powershell
cargo run -p adaforge -- projection-check models/gemma-4-12b-it-qat-q4_0/gemma-4-12B-it-QAT-Q4_0.gguf
```

Ce diagnostic charge `blk.0.attn_norm.weight` et `blk.0.attn_q.weight`, vérifie leurs dimensions et enchaîne nos deux kernels sur un même stream CUDA. Les poids sont transférés et les buffers alloués une seule fois pour le diagnostic. Le buffer de sortie RMSNorm devient directement l’entrée de matvec : aucun transfert hôte ni synchronisation hôte n’est inséré entre ces deux opérations. Ce n’est pas une fusion de kernels ni un CUDA Graph.

Quatre vecteurs sont contrôlés (zéro, motif, petite impulsion et pseudo-aléatoire déterministe). La référence calcule RMSNorm en FP64, arrondit son résultat en FP32 à la frontière de l’opérateur, puis accumule matvec en FP64. Elle ne réutilise aucun résultat GPU. La tolérance finale est `1e-4 + 1e-4 * abs(reference)` par sortie, valeurs non finies refusées.

Le temps affiché couvre les deux kernels et leur enchaînement, médiane de dix répétitions sur le dernier vecteur après échauffement, hors compilation, transferts et CPU. Il s’agit du début de la première couche : les projections K/V, normalisation des têtes, RoPE, attention et FFN restent à assembler.

## Projections Q/K/V

```powershell
cargo run -p adaforge -- qkv-check models/gemma-4-12b-it-qat-q4_0/gemma-4-12B-it-QAT-Q4_0.gguf
cargo run -p adaforge -- qkv-check models/gemma-4-12b-it-qat-q4_0/gemma-4-12B-it-QAT-Q4_0.gguf 5
```

La couche optionnelle est indexée à partir de zéro. Le diagnostic charge la normalisation d’entrée et les matrices Q/K/V, applique une seule normalisation et réutilise ce résultat GPU pour les trois projections. Il contrôle quatre vecteurs et toutes les sorties séparément pour Q, K et V, avec la même référence et tolérance que `projection-check`. Le temps couvre RMSNorm et les trois lancements de matvec.

Sur notre couche 0 : Q = 4096 × 3840, K/V = 2048 × 3840. Sur la couche 5 globale : Q = 8192 × 3840, K/V = 512 × 3840. Cette couche n’a pas de poids V distincts ; conformément à l’attention alternative Gemma 4, les poids K sont réutilisés pour produire V brut. Le diagnostic signale ce cas explicitement. Il lance actuellement deux calculs identiques et conserve deux buffers distincts pour K/V ; l’alias GPU destiné à éviter ce calcul redondant viendra avec la gestion des normalisations et du cache.

Les sorties sont les projections brutes : aucune normalisation par tête, rotation RoPE, attention ou cache KV pour le moment. Référence du cas V absent : https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modeling_gemma4.py (`Gemma4TextAttention`).

## Normalisation par tête Q/K

```powershell
cargo run -p adaforge -- qknorm-check models/gemma-4-12b-it-qat-q4_0/gemma-4-12B-it-QAT-Q4_0.gguf
cargo run -p adaforge -- qknorm-check models/gemma-4-12b-it-qat-q4_0/gemma-4-12B-it-QAT-Q4_0.gguf 5
```

La chaîne réutilise les projections et ajoute RMSNorm séparément sur chaque tête Q et K, avec leurs poids F32 respectifs. Le kernel RMSNorm existant traite un bloc de 256 threads par tête ; il n’est pas nécessaire d’introduire un nouveau kernel. Toutes les opérations restent sur le même stream CUDA et les intermédiaires restent sur GPU. Les largeurs sont lues dans les tenseurs `attn_q_norm.weight` et `attn_k_norm.weight`. Le diagnostic vérifie leur compatibilité avec les dimensions des projections et le regroupement des têtes.

La couche 0 comporte 16 têtes Q et 8 têtes K/V, de largeur 256 ; la couche 5 comporte 16 têtes Q et une tête K/V, de largeur 512. V reste brut dans ce diagnostic, y compris lorsque ses poids sont ceux de K : ses données sont conservées séparément avant la normalisation de K. La normalisation V sans échelle, requise par Gemma 4 avant l’attention, sera ajoutée dans une étape ultérieure.

Quatre vecteurs et toutes les sorties sont vérifiés face à une chaîne CPU indépendante, avec calculs FP64 et arrondis FP32 aux frontières des opérateurs (`atol = rtol = 1e-4`). Le temps couvre la chaîne complète, médiane de dix répétitions après échauffement. Les tests GPU synthétiques utilisent plusieurs têtes avec poids Q/K différents, des dimensions 32, 256 et 512, et le cas K/V partagé.

Cette étape n’applique pas encore RoPE et ne calcule pas l’attention. Référence de l’ordre des opérations : https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modeling_gemma4.py (`Gemma4TextAttention.forward`).

## Rotation positionnelle RoPE

```powershell
cargo run -p adaforge -- rope-check models/gemma-4-12b-it-qat-q4_0/gemma-4-12B-it-QAT-Q4_0.gguf
cargo run -p adaforge -- rope-check models/gemma-4-12b-it-qat-q4_0/gemma-4-12B-it-QAT-Q4_0.gguf 5
```

La chaîne ajoute la rotation NeoX après les normes Q/K. Les paires sont prises entre les deux moitiés d’une tête (indice i et i + largeur/2), pas entre des éléments adjacents. V reste brut et n’est jamais tourné.

Les métadonnées déterminent le type de couche, la base, la largeur et la limite de position. Les couches locales utilisent la base 10 000 ; les globales la base 1 000 000 et les facteurs du tenseur `rope_freqs.weight` (ou sa variante par bloc). Dans notre modèle global, 64 paires tournent normalement et 192 ont un facteur 1e30 qui atténue pratiquement leur rotation. Ces facteurs sont lus dans le modèle, pas imposés dans le kernel. Une dimension RoPE différente de la largeur de tête est refusée explicitement pour le moment.

Les fréquences inverses sont préparées en FP64 et transférées une seule fois. Le premier kernel utilise un angle et une trigonométrie FP64 pour garder une référence précise aux grandes positions ; les coefficients et résultats sont FP32. C’est un choix de justesse initiale, pas une optimisation de débit. Toutes les entrées et sorties des opérateurs restent sur GPU jusqu’à la validation finale.

Huit vecteurs/cas de position sont comparés à une chaîne CPU indépendante (FP64 avec frontières FP32), aux positions 0, 1, 127, 4095, milieu du contexte et près de sa limite, jusqu’à 262143 pour ce fichier. La position 0 inclut une entrée non nulle pour vérifier l’identité. Tolérance finale inchangée : `atol = rtol = 1e-4`. Les tests unitaires contrôlent aussi le sens de rotation, l’ordre des paires, la conservation de norme et le rejet des positions hors limites. Le temps correspond à la chaîne entière pour le dernier vecteur, médiane de dix répétitions après échauffement.

Références de sémantique et de représentation GGUF : https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modeling_gemma4.py et https://github.com/ggml-org/llama.cpp/blob/master/conversion/gemma.py (`Gemma4Model.generate_extra_tensors`). Notre implémentation ne copie pas les kernels de ces moteurs.

Le diagnostic `rope-check` conserve V brut pour isoler RoPE. Le diagnostic `vnorm-check` décrit ci-dessous ajoute la normalisation V. Pas encore de cache KV, de calcul d’attention, de FFN complet ni de génération de texte.

## Normalisation V sans échelle

```powershell
cargo run -p adaforge -- vnorm-check models/gemma-4-12b-it-qat-q4_0/gemma-4-12B-it-QAT-Q4_0.gguf
cargo run -p adaforge -- vnorm-check models/gemma-4-12b-it-qat-q4_0/gemma-4-12B-it-QAT-Q4_0.gguf 5
```

Cette étape termine la préparation des projections : Q/K normalisés et tournés par RoPE, V normalisé séparément par tête avec `v / sqrt(mean(v²) + epsilon)`. V n’a ni poids d’échelle appris ni rotation. Le nouveau kernel utilise la même réduction que RMSNorm, spécialisée à la compilation sans lecture de poids. Aucun vecteur d’échelles n’est transféré pour V.

La référence CPU utilise des poids unité pour exprimer mathématiquement l’absence d’échelle et accumule en FP64, avec arrondi FP32 après la projection brute. Elle ne lit aucun résultat GPU. Le diagnostic conserve les mêmes huit entrées/positions, tolérances et protocole de mesure que `rope-check` ; le temps couvre maintenant aussi le kernel V.

Les tests GPU contrôlent plusieurs têtes de largeurs 32, 256 et 512 avec des poids Q/K non unitaires et partagés avec V. Cela vérifie que V utilise bien sa projection brute, pas K déjà normalisé ou tourné. Les anciens diagnostics restent disponibles. Le cache KV et l’attention ne sont pas encore implémentés.

## Cache KV : stockage GPU

```powershell
cargo run -p adaforge -- kv-cache-check models/gemma-4-12b-it-qat-q4_0/gemma-4-12B-it-QAT-Q4_0.gguf
cargo run -p adaforge -- kv-cache-check models/gemma-4-12b-it-qat-q4_0/gemma-4-12B-it-QAT-Q4_0.gguf 5
```

Premier cache contigu FP32 pour une couche et une séquence, avec buffers K et V distincts, en ordre `[token][tête KV][canal]`. Les dimensions viennent des métadonnées Gemma ; la capacité de huit tokens est volontairement petite pour ce diagnostic. Le stockage accepte des ajouts unitaires ou par lots depuis des buffers GPU, sans copie des poids ni passage des données par le CPU dans l’opération d’ajout. Les copies GPU utilisent le pilote CUDA via cudarc, pas un kernel de copie réécrit.

Le test produit des K/V synthétiques déterministes (ce ne sont pas encore les sorties de Gemma), les transfère sur GPU puis les ajoute par lots de 3, 1 et 4 tokens. Après chaque lot, le préfixe complet est relu uniquement pour validation et comparé exactement au CPU. Il vérifie aussi le refus d’un neuvième token sans modification, la remise à zéro et un nouveau préfixe de deux tokens. Les tests GPU couvrent également les tailles d’entrée incorrectes et plusieurs géométries.

Les allocations restent fixes jusqu’à la destruction du cache. `reset` remet la longueur logique à zéro sans effacer physiquement les buffers ; seuls les éléments du nouveau préfixe sont exposés par le diagnostic. La longueur représente les écritures en file d’exécution : les consommateurs doivent utiliser le même stream. Une erreur de mise en file de copie invalide le cache, qui doit alors être recréé.

La commande `kv-cache-check` teste cette brique indépendamment de la chaîne de projection. Elle n’implémente ni attention, ni fenêtre glissante circulaire, ni pagination, ni partage entre requêtes, ni compression KV. La commande `kv-chain-check` ci-dessous connecte maintenant les sorties K après RoPE et V après normalisation. Le choix FP32 facilite la justesse initiale ; il ne préjuge pas du format optimisé final.

## Chaîne Gemma connectée au cache KV

```powershell
cargo run -p adaforge -- kv-chain-check models/gemma-4-12b-it-qat-q4_0/gemma-4-12B-it-QAT-Q4_0.gguf
cargo run -p adaforge -- kv-chain-check models/gemma-4-12b-it-qat-q4_0/gemma-4-12B-it-QAT-Q4_0.gguf 5
```

Ce diagnostic charge les poids réels du bloc demandé et traite huit vecteurs d’activations synthétiques aux positions successives 0 à 7. Chaque position enchaîne RMSNorm d’entrée, projections Q/K/V, normes par tête, RoPE Q/K et norme V sans échelle, puis ajoute directement K tourné et V normalisé au cache sur le même stream CUDA. Les poids, intermédiaires et buffers du cache sont alloués une seule fois pour la séquence. Il n’y a aucun transfert GPU → CPU → GPU pour alimenter le cache.

Après chaque ajout, le préfixe K/V complet est relu uniquement pour validation. Il est comparé à une référence CPU indépendante avec `atol = rtol = 1e-4`, et les éléments déjà présents sont aussi comparés exactement au snapshot précédent pour détecter tout écrasement. Le neuvième ajout doit échouer et préserver intégralement les données. Sur la couche 0, les huit contrôles successifs représentent 147456 valeurs vérifiées ; sur la couche 5, 36864.

Cette commande est un test d’intégration, pas un benchmark : elle effectue une exécution par position et des lectures de contrôle entre positions. Elle ne traite pas encore de texte, n’utilise pas de tokenizer et ne calcule pas l’attention. Les métadonnées des couches locales servent aux dimensions et à RoPE ; la politique de fenêtre glissante de l’attention n’est pas encore implémentée.

## Attention causale depuis le cache

```powershell
cargo run -p adaforge -- attention-check models/gemma-4-12b-it-qat-q4_0/gemma-4-12B-it-QAT-Q4_0.gguf
cargo run -p adaforge -- attention-check models/gemma-4-12b-it-qat-q4_0/gemma-4-12B-it-QAT-Q4_0.gguf 5
```

La chaîne produit les projections et leurs normalisations, tourne Q/K, ajoute K/V au cache puis calcule `softmax(QKᵀ * scale) V` directement depuis les buffers GPU. Pour Gemma 4, `scale = 1.0` (pas de division implicite par la racine de la largeur). Les têtes Q sont regroupées sur les têtes KV : 16/8 pour le bloc 0 et 16/1 pour le bloc 5. La fenêtre locale vient des métadonnées (1024 dans notre modèle) ; le bloc global n’a pas de fenêtre. Le masque ne rend visibles que les positions jusqu’au token demandé, même si le cache contient des tokens futurs.

Premier kernel FP32 de référence : un bloc de 256 threads par tête Q, réduction des produits scalaires et softmax stable par soustraction du score maximal. Les scores temporaires sont en mémoire partagée. Le backend refuse plus de 4096 tokens dans cette version, même si le modèle annonce un contexte plus long. Le cache doit commencer à la position zéro ; il ne s’agit pas encore d’un cache circulaire ni d’un kernel FlashAttention optimisé.

Le diagnostic vérifie la sortie de chaque position 0–7 contre une chaîne CPU indépendante, calculs FP64 avec frontières FP32 et `atol = rtol = 1e-4`. Les lectures CPU ne servent qu’à la validation : le calcul d’attention consomme les données résidentes. Il valide 32768 valeurs sur le bloc local et 65536 sur le global.

Les tests GPU isolés contrôlent une requête dans un cache contenant des tokens futurs, le regroupement des têtes, la fenêtre de largeur 1 et la limite 1024 sur 1025 tokens (ancienne valeur extrême pour détecter une erreur de borne), ainsi que des scores non nuls et de forte amplitude. Les calculs de référence testent aussi une échelle explicite différente de 1.

Ce n’est pas encore un bloc Transformer complet : projection de sortie de l’attention, normalisation post-attention, connexion résiduelle et FFN restent à construire. Les entrées restent des activations synthétiques avec les vrais poids Gemma. Aucun gain face à un moteur existant n’est établi à ce stade.

Référence de sémantique Gemma 4 : https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modeling_gemma4.py (`Gemma4TextAttention`, `eager_attention_forward`).

## Branche d’attention avec sortie et résiduel

```powershell
cargo run -p adaforge -- attention-branch-check models/gemma-4-12b-it-qat-q4_0/gemma-4-12B-it-QAT-Q4_0.gguf
cargo run -p adaforge -- attention-branch-check models/gemma-4-12b-it-qat-q4_0/gemma-4-12B-it-QAT-Q4_0.gguf 5
```

La chaîne étend l’attention causale avec `blk.N.attn_output.weight` (matrice Q4_0), `blk.N.post_attention_norm.weight` (échelle F32), puis l’ajout résiduel. L’ordre est : attention → projection de sortie → RMSNorm post-attention → addition de l’entrée brute du bloc. Le résiduel ne provient pas de l’entrée déjà normalisée.

Sur notre Gemma : la sortie d’attention de 4096 valeurs (bloc local) ou 8192 valeurs (bloc global) est projetée vers 3840. Les poids et buffers supplémentaires sont chargés une seule fois pour le diagnostic. La sortie d’attention est directement consommée sur GPU, puis normalisée et additionnée au résiduel sans transfert hôte intermédiaire.

Les huit positions 0–7 sont comparées à une référence CPU indépendante en FP64 avec arrondis FP32 entre opérateurs. La référence reprend toutes les étapes en amont ; aucune sortie GPU ne sert d’entrée à l’oracle CPU. La tolérance reste `atol = rtol = 1e-4` par sortie. Les contrôles antérieurs (Q/K/V, cache et attention) restent actifs. La branche finale représente 30720 valeurs vérifiées sur chaque bloc testé.

Le test GPU isolé de la sortie couvre une largeur 257 (dernier bloc CUDA partiel), des poids de normalisation non unitaires et une branche nulle, qui doit restituer exactement le résiduel. Le FFN et sa normalisation/résiduel restent à ajouter pour finir le bloc Transformer. Aucun texte n’est généré à ce stade.

Référence de l’ordre post-attention/résiduel : https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modeling_gemma4.py (`Gemma4TextDecoderLayer.forward`).

## Première couche dense assemblée (attention + FFN)

```powershell
cargo run -p adaforge -- layer-check models/gemma-4-12b-it-qat-q4_0/gemma-4-12B-it-QAT-Q4_0.gguf
cargo run -p adaforge -- layer-check models/gemma-4-12b-it-qat-q4_0/gemma-4-12B-it-QAT-Q4_0.gguf 5
```

La branche d’attention est suivie du FFN dense de Gemma :

1. RMSNorm pré-FFN sur la sortie après résiduel attention.
2. Projections Q4_0 gate et up (3840 → 15360 sur le modèle cible).
3. `GELU_tanh(gate) * up` : GELU avec approximation tanh, pas SiLU.
4. Projection down Q4_0 (15360 → 3840).
5. RMSNorm post-FFN.
6. Ajout du résiduel (entrée brute du FFN), puis multiplication par l’échelle finale de couche si présente dans GGUF.

Les activations et les poids résident sur GPU pendant les huit positions 0–7 ; les sorties d’attention alimentent directement le FFN sans retour CPU intermédiaire. Les références CPU reprennent tous les opérateurs avec accumulation FP64 et frontières FP32. Chaque couche diagnostiquée produit 30720 valeurs contrôlées avec la tolérance `atol = rtol = 1e-4`. Cette référence recalcule de grandes matrices et peut rendre le diagnostic sensiblement plus long que les étapes précédentes ; ce temps n’est pas le débit du moteur GPU.

Cette première implémentation est réservée au chemin dense Gemma 4 sans MoE ni embeddings auxiliaires par couche. Les poids sont lus depuis les tenseurs `ffn_norm`, `ffn_gate`, `ffn_up`, `ffn_down`, `post_ffw_norm` et l’échelle optionnelle `layer_output_scale`. La présence de MoE ou d’embeddings auxiliaires est refusée explicitement. L’activation GELU(tanh) est celle du chemin texte Gemma 4 ciblé.

Tests : valeurs analytiques de GELU, préservation du résiduel avec branche nulle, échelle finale non unitaire, FFN isolé et couche entière synthétique avec GQA, RoPE, cache et fenêtre causale. Les anciennes commandes gardent leur périmètre.

Ce jalon vérifie un bloc à la fois, avec poids réels et activations synthétiques. Il ne valide pas encore les 48 blocs enchaînés, les embeddings/tokenizer, les logits finaux ou la génération de texte. La comparaison avec les sorties d’un moteur de référence complet reste à faire.

Références de sémantique : https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/configuration_gemma4.py et https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modeling_gemma4.py (`Gemma4TextMLP`, `Gemma4TextDecoderLayer`).

## Embeddings Q6_K depuis les IDs de tokens

```powershell
cargo run -p adaforge -- embedding-check models/gemma-4-12b-it-qat-q4_0/gemma-4-12B-it-QAT-Q4_0.gguf
cargo run -p adaforge -- embedding-check models/gemma-4-12b-it-qat-q4_0/gemma-4-12B-it-QAT-Q4_0.gguf 2 42 1024 42
```

Le lecteur trouve la table `token_embd.weight` Q6_K et lit uniquement les lignes correspondant aux IDs demandés. Notre modèle contient 262144 lignes de 3840 valeurs. Par défaut, le diagnostic teste `[0, 1, 2, 255, 256, 1023, 262143, 1]` : début, frontières et fin de vocabulaire, avec répétition. L’ordre et les répétitions sont conservés. Les IDs hors vocabulaire sont refusés avant lecture des poids et avant initialisation CUDA ; maximum 256 IDs par appel.

La déquantification Q6_K utilise des blocs de 256 valeurs sur 210 octets : 128 octets de bits bas, 64 de bits hauts, 16 sous-échelles signées et une échelle FP16. Une référence CPU et un kernel CUDA décodent séparément ces données. Les échelles FP16 non finies sont refusées.

Chaque ligne est vérifiée brute puis multipliée par `sqrt(3840)` en FP32, conformément au chemin GGUF de Gemma (échelle ~61.967735). Les huit lignes donnent 61440 valeurs contrôlées avec `atol = rtol = 1e-6`. Sur le fichier cible, toutes les valeurs testées sont identiques CPU/GPU. Cela valide le décodage et la mise à l’échelle, pas encore une tokenisation.

La table complète n’est chargée ni en RAM ni sur le GPU. Les lignes sélectionnées sont compactées dans un buffer de diagnostic et déquantifiées dans le même ordre. Il ne s’agit pas encore d’un lookup dans une table entièrement résidente, d’une projection finale Q6_K ou d’un chemin connecté aux couches.

Les tests CPU construisent des codes 6 bits connus pour vérifier les 256 positions et les sous-échelles négatives ; les tests GPU ajoutent plusieurs lignes/blocs, échelles négatives et sous-normales. Les IDs fournis ici sont numériques : ils n’ont pas encore été obtenus à partir d’un texte par notre tokenizer.

Références de représentation et d’échelle GGUF : https://github.com/ggml-org/ggml/blob/master/src/ggml-quants.c (`dequantize_row_q6_K`) et https://github.com/ggml-org/llama.cpp/blob/master/src/models/gemma4.cpp (`build_inp_embd`).

## Embeddings connectés à la première couche

```powershell
cargo run -p adaforge -- embedding-layer-check models/gemma-4-12b-it-qat-q4_0/gemma-4-12B-it-QAT-Q4_0.gguf
```

Les huit IDs du diagnostic embeddings alimentent maintenant le premier bloc dense : lignes Q6_K réelles → déquantification et échelle sur GPU → première couche (attention, cache, sortie et FFN). Toutes les lignes sélectionnées sont déquantifiées une fois, puis chaque ligne résidente est copiée directement dans le buffer d’entrée de la couche sur le même stream. Les poids, modules et buffers restent vivants pendant le diagnostic.

Le CPU décode indépendamment ces lignes pour construire la référence complète, mais aucun de ces vecteurs CPU n’est transféré comme entrée à la couche GPU. Les contrôles intermédiaires et finaux restent actifs : huit positions et 30720 valeurs de sortie avec `atol = rtol = 1e-4`. Le test GPU du stockage résident vérifie aussi les copies de lignes dans un ordre différent, les répétitions et les bornes.

Ce diagnostic cible uniquement le bloc 0. Injecter directement un embedding au bloc 5 ne représenterait pas le chemin réel d’un modèle ; les diagnostics antérieurs de blocs isolés restent disponibles pour tester ce type de couche avec des données synthétiques.

Les IDs sont encore une séquence numérique choisie pour le test, pas le résultat d’une tokenisation de texte. Il reste à implémenter la tokenisation Gemma, à enchaîner les 47 blocs suivants et à produire les logits finaux. Ce jalon ne valide donc pas encore les réponses textuelles ni la qualité du modèle complet.


## Tokenisation du texte Gemma 4

```powershell
cargo run -p adaforge -- tokenize models/gemma-4-12b-it-qat-q4_0/gemma-4-12B-it-QAT-Q4_0.gguf "Bonjour, comment vas-tu ?"
```

Le tokenizer est implémenté en Rust dans le domaine, sans bibliothèque de moteur d’inférence. L’adaptateur GGUF charge uniquement le vocabulaire, les types et les rangs de fusion BPE ; aucun poids ni contexte CUDA n’est chargé. Il applique les fusions par rang, le remplacement des espaces par metaspace et le fallback UTF-8 sur les 256 octets.

La commande ajoute `<bos>` par défaut (`--no-bos` pour le désactiver). `--parse-special` interprète aussi les marqueurs de contrôle littéraux. Les tokens USER_DEFINED sont reconnus indépendamment de cette option, avec reclassement du marqueur Gemma `<|tool_response>` en contrôle pour correspondre au chargeur de référence. Le décodage affiché enlève seulement le BOS ajouté automatiquement. Le caractère littéral `▁` est décodé en espace : le diagnostic signale alors que l’aller-retour n’est pas exact. Limite du diagnostic : 64 Kio de texte UTF-8.

Il s’agit de texte brut, sans application du template de conversation, sans connexion automatique aux embeddings et sans génération. La commande affiche les IDs et le texte redécodé.

La validation externe utilise `llama-tokenize` installé localement, uniquement comme référence de test : llama.cpp build 10869, commit `30b6a755e`. Les fixtures couvrent 87 cas : texte vide, français, Unicode, emoji, espaces, LF/CRLF, tabulations, code, octet NUL et marqueurs spéciaux, avec différentes options BOS/contrôle. Le fichier de test est lu en mode binaire pour préserver CRLF et NUL sous Windows. Les IDs attendus sont comparés exactement au tokenizer Rust.

```powershell
cargo test -p adaforge-model-formats --test tokenizer_gemma -- --ignored
python scripts/verify_tokenizer.py models/gemma-4-12b-it-qat-q4_0/gemma-4-12B-it-QAT-Q4_0.gguf
```

La première commande utilise les fixtures livrées et le modèle local ; la seconde les régénère avec l’exécutable `llama-tokenize`, puis exécute la comparaison Rust. Source de sémantique : https://github.com/ggml-org/llama.cpp/blob/30b6a755e/src/llama-vocab.cpp.


## Texte connecté au premier bloc GPU

```powershell
cargo run -p adaforge -- text-layer-check models/gemma-4-12b-it-qat-q4_0/gemma-4-12B-it-QAT-Q4_0.gguf "Bonjour, comment vas-tu ?"
```

Cette commande compose le tokenizer Rust et le diagnostic embeddings/couche : texte brut → IDs avec BOS → lignes Q6_K sélectionnées → embeddings GPU mis à l’échelle → bloc 0 complet (attention causale, cache KV et FFN). Les embeddings de référence CPU servent aux contrôles ; ils ne sont pas utilisés comme entrées du calcul GPU. Les positions suivent exactement l’ordre des IDs, à partir de zéro. Le cache est dimensionné au nombre de tokens, sans ajout de tokens de remplissage.

Le diagnostic accepte de 1 à 32 tokens, BOS inclus. Un texte trop long est refusé avant chargement des poids et avant initialisation CUDA, sans troncature. Le texte vide correspond au BOS seul. Aucun template de conversation n’est appliqué et les marqueurs de contrôle littéraux ne sont pas interprétés. Les anciennes commandes embeddings et couche conservent leurs séquences de test par défaut.

Toutes les sorties du premier bloc sont comparées à une référence CPU avec accumulation FP64 et frontières FP32, `atol = rtol = 1e-4`. Ce diagnostic peut prendre du temps à cause de la référence CPU ; son temps total ne mesure pas les performances d’inférence. Il reste à enchaîner les 47 blocs suivants, la normalisation finale et les logits avant de générer du texte.


## Deux blocs consécutifs sur GPU

```powershell
cargo run -p adaforge -- text-stack-check models/gemma-4-12b-it-qat-q4_0/gemma-4-12B-it-QAT-Q4_0.gguf "Bonjour, comment vas-tu ?" 2
```

Le texte brut est tokenisé avec BOS, puis les embeddings Q6_K alimentent les blocs 0 et 1 dans cet ordre. Le backend possède maintenant un bloc résident réutilisable : poids, normes, fréquences RoPE, buffers attention/FFN et cache KV sont créés une fois et conservés pendant la séquence. Chaque token passe par les deux blocs. Le bloc 1 emprunte directement le buffer de sortie du bloc 0, sur le même stream CUDA. Chaque bloc dispose de son propre cache KV ; les positions vont de zéro au nombre de tokens moins un.

Le chargement des paramètres d’un bloc dense est partagé avec les diagnostics précédents. L’application conserve les paramètres et appelle un port sans type CUDA ; seul le backend connaît les buffers et kernels. Les géométries des deux blocs et des embeddings sont vérifiées avant initialisation CUDA. Un bloc résident refuse les positions sautées, répétées ou hors capacité ; une erreur d’exécution invalide son état.

Une référence CPU part des embeddings et recalcule les deux blocs sans lire leurs activations GPU. La sortie CPU du premier bloc, arrondie FP32, alimente le second bloc de référence. Chaque sortie GPU est comparée à cette chaîne CPU complète (`atol = rtol = 1e-4`). Les lectures GPU vers CPU servent au diagnostic, sans réinjection des activations CPU dans le calcul GPU.

Résultat sur la phrase ci-dessus : 30720 valeurs par bloc ; erreur maximale 0.00000763 / 0.00000572 et ratios 0.03289 / 0.03263 pour les blocs 0 / 1. Le test synthétique CUDA utilise deux blocs aux poids distincts, une fenêtre locale et une attention globale, et contrôle le refus des positions invalides, le dépassement sans mutation et la séparation des caches. Ce test synthétique ne valide pas encore la couche globale réelle de Gemma.

```powershell
cargo test -p adaforge-cuda-backend stack::tests -- --include-ignored
```

Le diagnostic reste limité à deux blocs et de 1 à 32 tokens, BOS inclus, sans troncature ni template de conversation. Les 46 blocs suivants, la normalisation finale et les logits restent à connecter. Les contrôles CPU dominent le temps de ce diagnostic ; aucun gain de débit d’inférence n’est établi.


## Six blocs et première attention globale

```powershell
cargo run -p adaforge -- text-stack-check models/gemma-4-12b-it-qat-q4_0/gemma-4-12B-it-QAT-Q4_0.gguf "Bonjour, comment vas-tu ?" 6
```

Le dernier argument `6` reproduit le jalon des blocs 0 à 5 ; `2` reproduit le jalon précédent. La commande accepte maintenant de 1 à 48 blocs, avec 48 par défaut. Chaque sortie reste comparée à la chaîne CPU complète depuis les embeddings, avec la même tolérance `atol = rtol = 1e-4`.

Sur le modèle cible, les blocs 0 à 4 utilisent l’attention locale : 16 têtes Q, 8 têtes KV et 256 canaux par tête. Le bloc 5 utilise l’attention globale : 16 têtes Q, une tête KV et 512 canaux par tête. La projection V réutilise les poids K, avec sorties et normalisations distinctes. Le RoPE global utilise la base 1000000 et les facteurs GGUF : 64 paires ordinaires et 192 paires à fréquence atténuée. Le rapport affiche ces géométries pour chaque bloc et confirme le partage des poids K/V.

Les buffers, sorties et caches ont leur propre taille par bloc ; seul le vecteur de largeur 3840 passe d’un bloc au suivant. Le test synthétique de transition change aussi les largeurs de tête, utilise K/V partagés et des fréquences atténuées. Les contrôles de positions et de caches restent actifs.

Ce diagnostic porte sur des préfixes de 1 à 32 tokens, BOS inclus. Il valide le calcul des premiers blocs et leur connexion, sans établir un gain de vitesse ni une équivalence avec les logits d’un moteur tiers. Les 42 blocs suivants et la tête de sortie restent à connecter.


Validation sur « Bonjour, comment vas-tu ? » : 184320 valeurs au total (30720 par bloc). Ratios maximaux pour les blocs 0 à 5 : 0.03289, 0.03263, 0.02609, 0.02532, 0.03953, 0.03517. La géométrie globale réelle du bloc 5 est confirmée : 16 Q / 1 KV, largeur 512, K/V partagés et 192 paires RoPE atténuées. Ces mesures de justesse ne constituent pas un benchmark d’inférence.


## Chaîne complète des 48 blocs

```powershell
cargo run --release -p adaforge -- text-stack-check models/gemma-4-12b-it-qat-q4_0/gemma-4-12B-it-QAT-Q4_0.gguf "Bonjour, comment vas-tu ?"
```

`text-stack-check` charge désormais les blocs 0 à 47 par défaut. Ajouter un dernier argument de 1 à 48 permet de vérifier un préfixe plus court. La limite de 32 tokens, BOS inclus, reste inchangée et aucun texte n’est tronqué. Le mode `--release` optimise notamment le calcul de référence CPU, tout en conservant les mêmes contrôles de sortie et les mêmes tolérances numériques.

Chaque bloc dispose de ses poids, buffers, fréquences RoPE et cache KV sur le GPU. Les huit blocs globaux (5, 11, 17, 23, 29, 35, 41, 47) sont sélectionnés par les métadonnées du modèle. La référence CPU propage ses propres sorties FP32 de bloc en bloc depuis les embeddings, sans réutiliser d’activations GPU. Les 48 sorties sont vérifiées avec `atol = rtol = 1e-4`.

Le lecteur GGUF conserve désormais son dernier index dans l’instance `GgufInspector`, créée avec `Default`. Les lectures de tenseurs réutilisent cet index au lieu de reparcourir le vocabulaire et les métadonnées pour chaque poids. Le cache est invalidé lorsque le chemin canonique, la taille ou la date de modification changent ; ses retours sont des copies indépendantes. Il ne conserve pas les poids ni le tokenizer. Les tests vérifient la réutilisation, l’indépendance des copies, le changement de fichier et les modifications de taille/date.

Ce jalon vérifie le corps du modèle. La normalisation finale, la projection vers le vocabulaire, le softcapping et le choix du token suivant restent à connecter ; la commande ne génère pas encore de réponse. Le temps total inclut le chargement, les compilations et tous les contrôles CPU, et ne représente pas le débit d’inférence.


Validation complète sur « Bonjour, comment vas-tu ? » (8 tokens avec BOS) : 1474560 valeurs vérifiées, 30720 par bloc. Le ratio maximal est de 0.81412 au bloc 35, sous la limite 1 ; l’erreur absolue maximale est de 0.00029945 au bloc 34. Au bloc 47, l’erreur maximale vaut 0.00001085 et le ratio 0.06572. La tolérance n’a pas été élargie. Cette validation porte sur cette séquence et ne garantit pas la même marge numérique pour d’autres textes ou contextes plus longs.


## Logits après les 48 blocs

```powershell
cargo run --release -p adaforge -- logits-check models/gemma-4-12b-it-qat-q4_0/gemma-4-12B-it-QAT-Q4_0.gguf "Bonjour, comment vas-tu ?"
```

Cette commande vérifie la chaîne des 48 blocs, puis calcule la tête de sortie pour la dernière position seulement. Le buffer GPU du bloc 47 alimente directement `output_norm.weight`, avec l’epsilon RMSNorm du modèle. La projection utilise `output.weight` si présent, sinon les poids liés `token_embd.weight`, au format Q6_K. Elle ne multiplie pas ces poids par l’échelle `sqrt(largeur)` des embeddings d’entrée. La matrice Q6_K complète est chargée sous forme quantifiée, sans déquantification en matrice FP32 intermédiaire.

Un kernel effectue le produit matrice-vecteur Q6_K, avec un warp par ligne, puis `cap * tanh(logit / cap)` ; le modèle cible utilise un cap de 30. Les logits bruts et après softcapping sont conservés séparément pour vérifier les deux étapes. Ce premier kernel privilégie la validation ; ses performances n’ont pas été comparées à celles d’un autre moteur.

La référence CPU commence toujours aux embeddings, propage ses activations à travers les 48 blocs, applique la RMSNorm finale puis recalcule chaque produit scalaire en FP64 avec des frontières FP32. Les 3840 valeurs normalisées, les 262144 logits bruts et les 262144 logits finaux sont contrôlés avec `atol = rtol = 1e-4`. Aucun intermédiaire CPU n’est utilisé comme entrée du calcul GPU. La comparaison avec les logits d’un moteur tiers reste à faire.

Sur la phrase ci-dessus (position finale 7), erreurs maximales / ratios : norme finale 0.00012028 / 0.37579 ; logits bruts 0.00007318 / 0.67380 ; logits finaux 0.00007308 / 0.67379. Tous les ratios sont sous la limite 1, sans élargir la tolérance. Cela valide ce texte de test et ne garantit pas la même marge pour tout autre préfixe.

Les tests ajoutent un produit Q6_K analytique, le softcapping, les entrées invalides et un test CUDA à 5 lignes et 2 blocs quantifiés par ligne, avec sous-échelles signées et échelle FP16 sous-normale. La limite de texte reste 32 tokens, BOS inclus ; aucun template de conversation ni sélection de token n’est appliqué. La commande ne génère pas encore de texte.

```powershell
cargo test -p adaforge-cuda-backend logits::tests -- --include-ignored
```

Référence de sémantique GGUF : https://github.com/ggml-org/llama.cpp/blob/master/src/models/gemma4.cpp (poids de sortie liés, normalisation et softcapping).


## Première génération et question en un tour

```powershell
cargo run --release -p adaforge -- ask models/gemma-4-12b-it-qat-q4_0/gemma-4-12B-it-QAT-Q4_0.gguf "Bonjour, comment vas-tu ?" 64
cargo run --release -p adaforge -- generate models/gemma-4-12b-it-qat-q4_0/gemma-4-12B-it-QAT-Q4_0.gguf "Le ciel est" 24
```

`ask` applique le sous-ensemble texte du template canonique présent dans ce GGUF : BOS, `<|turn>user\n`, message sans espaces de bord, `<turn|>\n<|turn>model\n<|channel>thought\n<channel|>`. Il prépare un seul échange, sans outils, système, multimodalité, historique ou pensée. Les marqueurs de cadrage sont interprétés séparément du message utilisateur pour préserver les marqueurs de contrôle littéraux qu’il contient. Il ne s’agit pas d’un interpréteur Jinja générique.

`generate` continue un texte brut, sans format de conversation. Ce mode peut donner une complétion peu pertinente pour une question : sur « Bonjour, comment vas-tu ? », AdaForge et le moteur de référence produisent tous deux une suite `1.000…`. Utiliser `ask` pour une question.

Les 48 blocs, leurs poids et leurs caches KV restent vivants pendant l’appel. Le prompt est traité token par token. Chaque nouveau token est sélectionné, déquantifié depuis la table Q6_K déjà sur GPU, puis traité à la position suivante. La table liée sert à la fois aux embeddings et à la tête de sortie, avec des mises à l’échelle distinctes. Les activations ne sont jamais recalculées sur CPU et les préfixes KV ne sont pas reconstruits. La sélection greedy est maintenant effectuée sur GPU ; seuls le résultat et son indicateur de validité reviennent sur CPU (8 octets). Les logits après softcap sont également calculés ; sélectionner sur les scores bruts respecte leur ordre monotone et évite les égalités FP32 créées par la saturation du tanh. À score brut égal, le plus petit ID gagne. Tout score non fini est refusé.

Limites de cette première version : 256 tokens de prompt, BOS et cadrage inclus ; de 1 à 128 nouveaux tokens (64 par défaut). Aucun texte n’est tronqué. L’arrêt est déclenché par `<eos>`, `<turn|>` ou `<|tool_response>`, ou par la limite de génération. Les marqueurs d’arrêt ne sont pas affichés. Si la limite coupe une séquence d’octets UTF-8, le décodage affiche un caractère de remplacement. La version cible explicitement le modèle dense Gemma 4 de 48 blocs avec poids de sortie liés en Q6_K ; une tête séparée est refusée.

La réponse est affichée à la fin de la génération, sans streaming. Chaque commande recharge le modèle ; les poids et les caches sont réutilisés entre tokens d’un même appel. Il ne s’agit pas encore d’une conversation interactive persistante.

Résultat avec `ask` sur la question ci-dessus : 20 tokens de prompt, 29 tokens générés, puis marqueur de fin. Réponse : « Bonjour ! Je vais très bien, merci de demander. Et vous, comment allez-vous aujourd'hui ? Comment puis-je vous aider ? »

Tests : choix greedy et égalités, scores invalides, lookup GPU Q6_K face au décodeur CPU pour plusieurs lignes/répétitions, tête résidente face aux logits CPU, caches incrémentaux, cadrage Gemma et fin UTF-8 partielle. La comparaison externe ci-dessous utilise `llama-completion` uniquement comme oracle de test, avec le même prompt et température zéro, sans intégrer ce moteur à AdaForge.

```powershell
python scripts/verify_generation.py models/gemma-4-12b-it-qat-q4_0/gemma-4-12B-it-QAT-Q4_0.gguf
```


Comparaison externe réussie sur cet échange : texte de réponse identique à `llama-completion` local (température zéro), après retrait de son marqueur d’affichage `[end of text]`. L’arrêt du modèle, une limite à deux tokens, la stabilité du préfixe et le refus des limites invalides sont vérifiés par le script. Cette comparaison porte sur un échange ; elle ne remplace pas une évaluation sur plusieurs tâches et contextes.


## Conversation interactive persistante

```powershell
cargo run --release -p adaforge -- chat models/gemma-4-12b-it-qat-q4_0/gemma-4-12B-it-QAT-Q4_0.gguf
```

Un dernier argument de 1 à 128 règle le maximum de tokens par réponse (64 par défaut). Attendre « Chat prêt », puis saisir un message par ligne. Les commandes `/quit` ou `/exit` terminent la session ; `/help` affiche l’aide ; `/reset` efface l’historique et remet les caches KV à zéro en conservant les poids, kernels et buffers alloués. Une fin d’entrée standard quitte aussi proprement.

Le modèle et le tokenizer sont chargés une seule fois. Le port `GenerationSession` de l’application masque complètement les ressources CUDA. L’adaptateur possède la tête de sortie, la table liée Q6_K, les 48 blocs et leurs caches indépendants. Les paramètres CPU temporaires du chargement sont libérés une fois la session créée.

L’historique conserve les IDs exacts : cadrage Gemma, messages, tokens générés et fermeture de tour. Le premier tour contient un seul BOS ; les suivants n’en ajoutent pas. Le backend vérifie que le nouveau prompt prolonge exactement son préfixe KV, puis calcule uniquement les tokens nouveaux. Même le dernier token produit avant une limite est consommé dans les caches pour rendre le tour suivant cohérent. Une réponse tronquée est conservée comme réponse partielle et refermée pour le tour suivant. Le format reste le sous-ensemble texte sans outils ni pensée ; aucun interpréteur Jinja complet n’est ajouté.

La capacité initiale est volontairement limitée à 512 tokens pour l’ensemble de l’historique, avec réservation du budget maximal de la prochaine réponse. Un message trop long est refusé avant modification des caches ou de l’historique. Le chat invite à utiliser `/reset` ; il ne supprime pas silencieusement d’anciens messages. Les erreurs CUDA invalident la session et nécessitent de relancer la commande. La sortie reste affichée à la fin de chaque réponse, sans streaming.

Test fonctionnel : après « Mon prénom est Luc. », « Quel est mon prénom ? » reçoit « Votre prénom est Luc. ». Après `/reset`, la question « Bonjour, comment vas-tu ? » retrouve la réponse du test en un tour. Le test GPU synthétique contrôle les préfixes KV exacts, le dernier token en cache, le refus d’un préfixe modifié, les limites, l’arrêt immédiat et la remise à zéro des positions/caches.

```powershell
cargo test -p adaforge-cuda-backend stack::tests -- --include-ignored
python scripts/verify_chat.py models/gemma-4-12b-it-qat-q4_0/gemma-4-12B-it-QAT-Q4_0.gguf
```

Le script compare la deuxième réponse issue du cache persistant à un recalcul complet du même transcript dans `llama-completion` local. Il vérifie aussi qu’un message refusé pour dépassement de contexte préserve l’historique et que `/reset` rétablit le résultat attendu. Le moteur de référence reste uniquement un outil de test.


## Affichage progressif et budget en cours de chat

Le chat affiche maintenant les fragments de texte dès leur génération et vide le buffer du terminal à chaque fragment. Le décodeur attend un caractère UTF-8 complet avant de l’émettre ; si la limite coupe une séquence d’octets, le caractère de remplacement est ajouté à la fin. Les tokens générés et l’historique restent identiques au mode sans streaming.

Le budget par défaut du chat est désormais 128 tokens par réponse. `/tokens` affiche le budget ; `/tokens 64` ou `/tokens 128` le change sans recharger le modèle ni effacer l’historique. La plage autorisée reste de 1 à 128. Un budget incompatible avec le contexte restant est refusé sans modifier l’état. `/reset` conserve le budget choisi. Les commandes `ask` et `generate` gardent leur défaut de 64 tokens.

Les tests passent pour les caractères UTF-8 coupés entre tokens, les callbacks de génération, les changements de budget et leur refus, la conservation de l’historique et l’affichage d’une réponse limitée à deux tokens. La comparaison du deuxième tour avec le moteur de référence et le test de remise à zéro passent aussi avec le streaming activé.


## Mesures de vitesse et comparaison CUDA

Le chat et `ask` affichent désormais les mesures du backend après chaque réponse : délai du premier token, débit de génération, nombre de nouveaux tokens de prompt et taille du préfixe réutilisé. La tokenisation, le cadrage, le chargement et la compilation initiale sont exclus de ces chronométrages. En chat, les écritures et callbacks entre tokens peuvent contribuer au temps écoulé. Moins de deux tokens ne permet pas de calculer un débit de décodage.

```powershell
cargo run --release -p adaforge -- bench models/gemma-4-12b-it-qat-q4_0/gemma-4-12B-it-QAT-Q4_0.gguf "Bonjour, comment vas-tu ?" 24 5
```

`bench` charge le modèle une fois, effectue une génération d’échauffement, puis remet les caches KV à zéro avant chacune des mesures. Les poids et kernels restent chargés. Il utilise le même cadrage que `ask`, sans affichage progressif pendant les mesures. Les lignes `BENCH_SAMPLE` contiennent les IDs exacts et les durées en JSON ; les lignes lisibles affichent les débits.

Définitions : le prefill traite uniquement les tokens de prompt nouveaux et synchronise le stream en fin de phase. Le premier token est daté après sélection greedy et récupération de son ID, avant consommation de son embedding. Le débit de génération vaut `(N-1) / (date du dernier token sélectionné - date du premier)`, sans inclure le prefill ni le calcul du premier token. La durée totale attend aussi la fin des kernels de consommation du dernier token et peut donc dépasser la somme prefill/décodage. Un arrêt immédiat n’a pas de délai de premier token généré.

La version système de `llama.cpp` utilisait Vulkan. Une distribution CUDA officielle distincte, build 11669 / commit `620d2225c` (0.7.0-dev), est installée dans `tools/llama-b11669-cuda12.4/bin`. Elle détecte `CUDA0: NVIDIA GeForce RTX 4090`. Les archives binaires CUDA 12.4 et leur runtime proviennent de la publication officielle https://github.com/ggml-org/llama.cpp/releases/tag/b11669 ; leurs SHA-256 et URL sont conservés dans `tools/llama-b11669-cuda12.4/provenance.json`. Le PATH et la version système ne sont pas modifiés.

```powershell
python scripts/install_llama_cuda.py
python scripts/compare_performance.py models/gemma-4-12b-it-qat-q4_0/gemma-4-12B-it-QAT-Q4_0.gguf
python scripts/compare_performance.py models/gemma-4-12b-it-qat-q4_0/gemma-4-12B-it-QAT-Q4_0.gguf --fp32
```

Le script lance les moteurs successivement, sur le même GGUF : 20 tokens de prompt identiques, 24 tokens de sortie identiques, sélection greedy, contexte 512, une requête à la fois, un échauffement et cinq répétitions. Pour `llama.cpp`, il démarre un serveur local temporaire sur une adresse de boucle locale et un port libre, avec `CUDA0`, toutes les couches GPU et ajustement automatique de chargement désactivé. Il vérifie le journal `offloaded 49/49 layers to GPU` et CUDA0, puis termine son propre serveur après les mesures. Les 49 couches annoncées comprennent les 48 blocs et la sortie.

Par défaut, la référence utilise ses caches FP16 et Flash Attention automatique ; AdaForge utilise actuellement FP32 et son attention initiale. Ce protocole mesure donc les implémentations avec leurs choix respectifs. `--fp32` impose FP32 et désactive Flash Attention côté référence pour une seconde observation. Le cache de prompt est désactivé et le nombre de tokens de prompt recalculés est vérifié dans chaque réponse. Les tokens générés doivent correspondre exactement avant toute comparaison.

Le débit de la référence est calculé à partir des dates de réception des tokens sur HTTP/SSE local, avec la même formule `(N-1)/durée`. Son délai de premier token inclut HTTP/SSE ; celui d’AdaForge est interne au backend. Ces latences ont donc des périmètres légèrement différents. La VRAM est échantillonnée via `nvidia-smi` pour la carte entière et inclut le bureau et les autres applications ; ce n’est pas une allocation exclusive au processus. Les valeurs brutes, commandes, minima, maxima et médianes sont conservés dans `benchmarks/comparison-cuda-native.json` et `benchmarks/comparison-cuda-fp32.json`.

Premier résultat avec caches FP16/Flash Attention automatique côté référence : médiane AdaForge 44.78 tok/s (44.21–45.82), référence CUDA 80.37 tok/s (79.59–82.02). Délais médians du premier token : 371.8 ms et 52.0 ms, avec les périmètres distincts ci-dessus. Le journal confirme 49/49 couches sur GPU. Sur cet échange court, AdaForge est donc encore plus lent en génération. Ces cinq mesures sur un seul prompt ne constituent pas une évaluation générale, ni une preuve de supériorité d’un moteur sur toutes les charges.


Le second profil, KV FP32 et Flash Attention désactivée côté référence, passe aussi la vérification GPU et l’égalité des tokens : médianes 46.18 tok/s pour AdaForge et 57.44 tok/s pour `llama.cpp` CUDA. La référence varie ici de 53.30 à 78.08 tok/s ; cette dispersion et son prefill inhabituellement lent empêchent d’en tirer une conclusion générale sur la latence. Le profil natif FP16/Flash Attention automatique reste la comparaison principale.


## Première passe d’optimisation RTX 4090

La cible reste au moins **100 tokens/s** en génération et une comparaison favorable à `llama.cpp` CUDA. Elle n’est pas encore atteinte. Les résultats initiaux ci-dessus précèdent cette passe ; les rapports de mesures conservent les échantillons et les IDs exacts.

Les matrices Q4_0 traitent quatre blocs quantifiés par warp avec lectures groupées et accumulateurs indépendants. Les projections Q/K/V partagent un lancement ; une projection K/V commune est calculée une seule fois. Les projections gate/up du FFN partagent également un lancement. La tête Q6_K réutilise les octets quantifiés entre plusieurs canaux. La sélection greedy utilise deux réductions GPU : les 262144 scores ne sont plus rapatriés à chaque token. Les égalités conservent le plus petit ID et tout score non fini reste refusé.

La RMSNorm réduit les synchronisations en conservant l’arbre de réduction FP32 précédent, vérifié bit à bit sur les dimensions testées. L’attention calcule plusieurs scores de tokens par bloc, avec le même arbre de somme, puis conserve le softmax stable et l’accumulation des valeurs dans l’ordre des positions. Aucun changement de quantification, de précision du cache ou de tolérance de validation n’est introduit.

Une session capture les embeddings et les 48 blocs dans un graphe CUDA réutilisable. L’ID du token et sa position sont mis à jour sur GPU avant chaque rejeu ; les caches KV restent propres à chaque bloc. La tête et la sélection du token suivent le graphe. Un stream privé ordonne toutes les opérations. Le graphe est détruit avant ses buffers ; une erreur invalide la session. La présence de la variable `ADAFORGE_DISABLE_GRAPHS` permet de tester le chemin direct. Ce choix concerne le backend CUDA ; le domaine et les ports applicatifs restent indépendants de CUDA.

Validation : tests du workspace, 24 tests GPU explicites, comparaison exacte graphe/chemin direct avec tokens et positions variables puis reset, sélection greedy avec égalités et scores non finis, RMSNorm identique bit à bit à sa version précédente. Sur le modèle complet et le prompt « Bonjour, comment vas-tu ? », les 48 blocs et 262144 logits passent la référence CPU avec les tolérances initiales `atol=rtol=1e-4` : ratio maximal 0.33389 pour la norme finale, 0.50711 pour les logits bruts et 0.50721 après softcap (limite 1).

Le profil Nsight Systems initial attribuait environ 79 % du temps des kernels aux matvec Q4_0. Ce profil comprend échauffement, prefill et décodage ; il sert à identifier le travail dominant, pas à annoncer un débit d’inférence. Les compteurs matériels Nsight Compute sont indisponibles avec les permissions actuelles ; aucune configuration globale du pilote n’a été modifiée.

Le rapport `benchmarks/graph-vs-direct.json` compare cinq répétitions après échauffement : 67.42 tok/s avec graphe contre 58.72 sans graphe, avec les mêmes IDs. Ce test mesure uniquement le bénéfice du graphe dans la version optimisée. Le chargement et la compilation initiale sont exclus du débit ; leur optimisation, le prefill par lots et les contextes longs restent à traiter.

```powershell
python scripts/compare_optimization.py
python scripts/compare_performance.py models/gemma-4-12b-it-qat-q4_0/gemma-4-12B-it-QAT-Q4_0.gguf
```

Le premier script nécessite l’exécutable initial conservé localement dans `tools/optimization-baseline/adaforge.exe`. Il mesure successivement avant/après/avant, avec un échauffement puis cinq répétitions par processus, remet les caches à zéro et exige des IDs identiques. Le second compare la version courante au serveur CUDA de référence, avec son cache FP16 et Flash Attention automatique. Ces résultats portent sur un prompt court et une seule requête ; ils ne garantissent pas le même débit sur toute conversation.


Résultat après cette passe : `optimization-before-after.json` donne 44.38 tok/s avant, 68.62 après et 42.58 au second passage de l’ancien exécutable. La nouvelle comparaison CUDA (`comparison-cuda-native.json`) donne 66.15 tok/s pour AdaForge (60.98–69.54) contre 80.54 pour llama.cpp (73.97–81.10), avec des IDs identiques et 49/49 couches GPU vérifiées. Le précédent rapport natif est conservé dans `comparison-cuda-native-before-optimization.json`. AdaForge reste donc derrière la référence et sous la cible de 100 tok/s. Les scripts de validation d’un tour et du chat persistant passent après optimisation : réponse exacte, limites, historique, dépassement refusé et reset.

La spécialisation reste explicitement RTX 4090 / compute capability 8.9. Le futur moteur pour une A2000 constitue une cible distincte : le tokenizer, le GGUF et les ports pourront être réutilisés, mais les choix de kernels, de cache et de mémoire devront être mesurés pour cette autre carte. Les prochaines pistes pour la 4090 sont le réarrangement interne des poids Q4_0 sans perte, les accès mémoire et l’occupation des kernels, puis un prefill par lots avec une stratégie distincte du décodage. Un gain de prefill ne doit pas être présenté comme une hausse du débit de génération.

Le profil final avec traçage des nœuds du graphe (`profile-optimized-summary.json`) confirme la priorité : 74.6 % du temps des kernels dans les projections Q4_0 (37.7 % gate/up, 26.1 % autres matvec, 10.8 % Q/K/V), devant RMSNorm 10.2 %, tête Q6_K 6.5 % et attention 5.5 %. Il comprend échauffement et prefill ; il ne prouve pas à lui seul une saturation de la bande passante mémoire. Le prochain essai prioritaire est donc un rangement interne des poids Q4 adapté aux lectures des warps, sans modifier les valeurs du modèle.


## Deuxième passe : essais Q4 mesurés et écartés

Les sources de production ont été rétablies exactement à la version validée avant ces essais. Aucun gain supplémentaire reproductible n’est revendiqué. Les mesures brutes et les IDs sont conservés dans les rapports JSON ; chaque candidat avant/après/avant utilise cinq répétitions après échauffement, le même prompt, 24 tokens générés et exige des IDs identiques.

| Essai | Avant | Candidat | Avant répété | Décision |
|---|---:|---:|---:|---|
| Q4 scales/quant bytes separated per row | 68.13 | 66.24 | 68.04 | Écarté |
| Hardware FP16 to FP32 scale conversion | 66.93 | 68.45 | 69.10 | Écarté |
| Gate and up computed together per warp | 64.04 | 58.38 | 67.91 | Écarté |
| Exact FP32 nibble conversion with LOP3 | 67.51 | 65.90 | 67.01 | Écarté |

Débits en tokens/s. Le balayage séparé de 1, 2, 8 et 16 warps par bloc n’a pas montré de gain net justifiant le remplacement des quatre warps actuels. La validation CPU de la projection passe pour le rangement des poids, la conversion matérielle des échelles et le candidat LOP3 ; celle de la couche complète passe pour gate/up conjoint. Cela ne suffit pas à justifier leur adoption sans gain de bout en bout. Après restauration, les tests du workspace et les 24 tests GPU explicites passent.

Pistes apportées pour la suite et limites vérifiées :

- La déquantification half2/LOP3 mérite un essai distinct du candidat FP32 testé ici. Une conversion des poids déquantifiés ou des produits en FP16 peut changer les arrondis ; elle nécessite une validation numérique et de génération adaptée. Le format QAT ne prouve pas que toutes les modifications du calcul conservent les résultats.
- Une lecture vectorielle de 16 octets exige un alignement convenable : le pas GGUF natif de 18 octets par bloc ne fournit pas cet alignement pour chaque bloc. Le rangement expérimental de cette passe séparait les échelles et les données ; il n’a pas accéléré la génération.
- Le cache KV local FP32 des 40 couches, à fenêtre 1024, représente `40 * 2 * 1024 * 8 * 256 * 4 = 671088640` octets, soit 640 Mio. Même à capacité actuelle 512, il représente 320 Mio. Il ne tient pas intégralement dans les 72 Mio de L2 de la RTX 4090. Les fenêtres de persistance CUDA donnent une priorité, pas une garantie de résidence : https://docs.nvidia.com/cuda/archive/13.1.1/cuda-programming-guide/04-special-topics/l2-cache-control.html
- Le partage de la matrice K/V globale est déjà exploité pour éviter une deuxième projection. K après RoPE et V normalisé sans rotation restent deux valeurs différentes ; leurs caches ne peuvent pas être simplement confondus.
- La sélection actuelle est greedy, entièrement sur GPU, et ne renvoie que 8 octets. Un top-k=64 changerait le besoin fonctionnel. Fusionner davantage la tête Q6_K avec la réduction est une autre piste ; sa part mesurée est de 6.5 % du temps des kernels dans le profil court actuel.
- Google documente un drafter MTP pour Gemma 4 12B. Son intégration exige notamment la vérification de plusieurs tokens en une passe et la gestion des acceptations/rejets et du cache. Les 250–300 tokens/s suggérés ne sont pas vérifiés ici : https://ai.google.dev/gemma/docs/mtp/mtp
- Le rapport bande passante théorique / taille du modèle est un modèle de plafond simplifié, pas une mesure de bande passante effectivement utilisée ni une décomposition de la latence de llama.cpp. Les pourcentages d’utilisation et les gains promis doivent être mesurés sur la charge réelle.

Synthèse structurée : `benchmarks/optimization-second-pass.json`. Les fichiers d’expérimentation sont archivés localement sous `tools/optimization-repack-experiment` et les sources initiales sous `tools/optimization-repack-baseline`.


## Relecture d’Imp : partage de la dimension d’entrée du FFN

L’étude du commit `50dd22ba57a4b67c4565eff3ac25fb76ad83fb4a` d’Imp a identifié une répartition absente de notre premier balayage : plusieurs warps coopèrent sur une seule ligne, au lieu d’affecter une ligne à chaque warp. Notre propre kernel `q4_matvec_kparallel` applique ce principe à la projection down du FFN, dont l’entrée a 15360 canaux. Les poids Q4_0 et les activations FP32 sont conservés ; quatre sommes partielles sont réduites dans le même bloc. Les autres projections gardent leur chemin précédent. Aucune bibliothèque ou source Imp n’est intégrée au moteur.

Mesure avant/après/avant : 66.87 / 70.82 / 67.74 tok/s avec les mêmes IDs (`benchmarks/optimization-kparallel-down.json`). L’extension à toutes les projections n’ajoute qu’environ 2 % dans un second essai et n’est pas retenue à ce stade. Les contrôles des 48 blocs et 262144 logits passent avec les tolérances inchangées : ratios 0.29132 pour la norme finale, 0.42836 pour les logits bruts et 0.42835 après softcap. Les tests du workspace, les 25 tests GPU explicites et Clippy passent. Le nouveau test GPU couvre notamment les dimensions 32, 96, 384 et 15360, les échelles nulles/négatives/subnormales et plusieurs amplitudes d’entrée.

La relecture détaille aussi le chemin Q4_0/Q8_1 `dp4a` d’Imp, ses fusions et les différences avec la RTX 5090 : [notes de relecture](docs/imp-review.md). La cible de 100 tok/s reste ouverte.

```powershell
python scripts/compare_optimization.py --baseline tools/optimization-repack-baseline/adaforge.exe --output benchmarks/optimization-kparallel-down-repeat.json
```

Le script accepte maintenant un exécutable de référence et un chemin de rapport explicites pour reproduire les comparaisons de chaque étape sans écraser les autres rapports.


Comparaison CUDA finale après validation : médiane AdaForge **71.73 tok/s** (70.03–72.52), llama.cpp CUDA **78.52 tok/s** (76.44–80.28). Même prompt, mêmes 24 IDs générés, un échauffement puis cinq répétitions, cache de prompt désactivé et journal 49/49 couches GPU vérifié. Le rapport courant est `benchmarks/comparison-cuda-native.json` ; celui de la première passe est conservé dans `benchmarks/comparison-cuda-native-first-pass.json`. Les scripts de validation de génération et de chat passent après le changement. AdaForge reste sous la référence et sous 100 tok/s.


## Prototype expérimental : activations Q8 et dp4a dans le FFN

Un chemin optionnel est disponible pour tester les trois projections du FFN avec des activations INT8 par groupes de 32 et des échelles FP32. Les poids restent Q4_0 ; le chemin par défaut conserve FP32. Le probe contrôle le calcul face à une référence CPU quantifiée et rapporte séparément l’écart à FP32. Le FFN quantifié complet possède aussi un test CPU/GPU indépendant. Les 26 tests GPU du backend, le test GPU de l’exemple, les tests du workspace et Clippy passent.

Les premières comparaisons donnent seulement un petit gain : 73.37 tok/s contre 70.90 / 71.34 en FP32 avec down partagé entre warps, puis 74.71 contre 73.69 / 73.21 avec down à une ligne par warp. Ces chiffres ne justifient pas un changement de précision par défaut. Les IDs générés diffèrent entre les précisions, et la qualité générale reste à évaluer. La cible de 100 tok/s n’est pas atteinte.

[Protocole, activation manuelle et limites du prototype](docs/q8-experiment.md). `scripts/compare_q8.py` produit `benchmarks/q8-experimental.json` et rapporte explicitement la divergence des IDs. La comparaison CUDA validée refuse le mode expérimental pour éviter de mélanger les résultats.


## Catalogue des prochaines optimisations

Les tuiles Q4 synchrones sont maintenant sélectionnées par défaut pour le FFN Gemma 3840/15360, sans duplication des poids GPU. Une qualification ABBA donne 72,27/72,45 contre 69,74/69,81 tok/s sur le prompt testé (+3,7 %), avec IDs identiques ; ce résultat ne garantit pas le même gain sur tous les contextes. L'override historique `ADAFORGE_EXPERIMENTAL_TILED_GATE=0` permet de revenir au natif. Les variantes `cp.async` testées sont correctes mais plus lentes et restent dans le banc de recherche. Voir [résultats et protocole](docs/optimization-async-and-abba.md).

Le [plan consolidé](docs/optimization-plan.md) regroupe les pistes AdaForge, Imp, Marlin et Ada, les essais écartés et leurs dépendances. La [lecture détaillée de Marlin](docs/marlin-review.md) décrit le kernel et les différences avec notre GGUF à groupes de 32. Les sources de référence sont conservées sous `tools/marlin-reference` sans ajout de dépendance au moteur.

## Optimisations d'octobre 2026

Génération ~95 tok/s sur prompt court (79 avant), ~200 tok/s quand la réponse reprend le prompt (décodage spéculatif greedy), ~83 tok/s à 7 600 tokens de contexte (26 avant). Premier token ~40 ms, prefill de 7 620 tokens en ~16,5 s, contexte du chat porté à 10 240 tokens. Toutes ces optimisations gardent des résultats bit-à-bit identiques. Détails, essais écartés et variables d'environnement : [docs/optimization-2026-10.md](docs/optimization-2026-10.md).
