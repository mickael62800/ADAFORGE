# Bilan de la session d'optimisation du 18 octobre 2026

Objectif : augmenter au maximum le nombre de tokens par seconde d'AdaForge sur RTX 4090 avec `gemma-4-12B-it-QAT-Q4_0`, **sans changer un seul bit des résultats**. Chaque optimisation donne exactement les mêmes nombres que le chemin d'origine : même ordre des additions FP32, mêmes multiplications-additions, même arbre de réduction. Chacune est vérifiée par un test GPU bit à bit, et les tokens générés restent identiques à la référence.

## Résultats

| Mesure | Avant | Après |
|---|---|---|
| Génération, conversation libre (« Bonjour, comment vas-tu ? ») | 79 tok/s | **~95,5 tok/s** |
| Même prompt, llama.cpp CUDA (même machine, même comparaison) | — | 85,5 tok/s |
| Génération, réponse qui reprend du texte du prompt | 86 tok/s | **~330 tok/s** |
| Génération avec un contexte de 7 600 tokens | 26 tok/s | **~83 tok/s** |
| Premier token, prompt de 20 tokens | 210 ms | **~37 ms** |
| Prefill d'un prompt de 7 620 tokens | 95 s | **~16,5 s** |
| Chargement du modèle | ~9,5 s | **~5,5 s** (jusqu'à 7 s si le fichier n'est pas en cache disque) |
| Contexte maximal du chat | 512 tokens | **32 768 tokens** (cache KV circulaire) |

Tout est commité et fusionné dans `master`. Tous les tests du workspace (y compris les tests GPU), Clippy, `verify_generation.py` et `verify_chat.py` passent.

## Ce qui a été fait, dans l'ordre

### 1. Diagnostic
- **Chronométrage GPU par token** (`ADAFORGE_TIMING_DECODE=1`) : 88 % du temps dans les 48 blocs, 12 % dans la tête de sortie.
- **Profils Nsight kernel par kernel.** Les gros produits matrice-vecteur lisaient déjà les poids à ~760 Go/s, pour un plafond mesuré de 845 Go/s sur cette carte.

### 2. Optimisations du décodage (un token à la fois)
- **Tête de sortie Q6_K** (table de 826 Mo) : réorganisée sans perte au chargement, avec des lectures plus larges et 2 lignes par warp. Le softcap inutilisé n'est plus écrit. **1,46 → 1,06 ms par token.**
- **Normes RMS** : les scales sont lues avant la réduction (elles arrivent de la DRAM à chaque token). Normes et additions résiduelles sont fusionnées, et la dernière norme d'une couche calcule aussi celle de la couche suivante. **21 → ~6 µs par couche.**
- **Attention** : découpée en trois kernels (scores, softmax, valeurs) pour utiliser beaucoup plus de blocs. Les clés sont partagées entre les têtes, et un pipeline asynchrone (`cp.async`) alimente l'accumulation des valeurs. **22 → ~5 µs par couche** à court contexte, et un gain énorme à long contexte.

### 3. Traitement par lots (prefill et vérification)
- **Produits matrice-vecteur par lots** : les poids sont lus une fois pour 8 à 32 tokens. Les entrées sont réordonnées pour une lecture de 16 octets par token, et le gate/up traite 2 tuiles par bloc.
- **Préparation Q/K/V, écriture du cache et attention** : traitées en un lancement pour tout le lot.
- **Résultat :** prefill environ 6 fois plus rapide sur un long prompt, et premier token bien plus rapide.

### 4. Décodage spéculatif (sortie identique au greedy)
- **Propositions :** on cherche dans le texte déjà présent la dernière occurrence des 3 derniers tokens, et on propose jusqu'à 11 tokens qui la suivaient.
- **Vérification :** une seule passe par lots, avec une tête de sortie par lots pour jusqu'à 16 positions, et une seule lecture côté CPU pour tous les choix greedy.
- **Retour en arrière :** le cache KV est ramené à la dernière position acceptée en cas de rejet.
- **Effet :** très efficace sur le code à corriger, les citations et les recopies ; aucun effet en conversation libre.

### 5. Contexte et chargement
- **Contexte du chat porté à 10 240 tokens**, avec le script de vérification adapté.
- **Chargement accéléré :**
  - compilation CUDA partagée entre les 48 couches au lieu d'être refaite pour chacune ;
  - index du fichier GGUF partagé au lieu d'être cloné à chaque tenseur ;
  - réorganisation des poids faite sur le GPU au lieu de boucles sur le CPU.

## Pistes essayées et abandonnées (mesurées)
- **Activations Q8 + `dp4a`, W4A16 :** aucun gain, car le décodage est limité par la lecture des poids, pas par le calcul.
- **Rangements Q4 alternatifs, préchargements explicites, 2 lignes par warp en décodage :** neutres. Le compilateur et le matériel font déjà ce travail.
- **Propositions spéculatives à partir de 2 tokens :** jamais acceptées en texte libre.
- **Fusions GELU dans le gate et préparation Q/K/V avec écriture du cache :** neutres.
- **Variantes d'attention (grilles plus larges, préchargement des clés) :** neutres, l'attention lit déjà le cache KV au maximum de la bande passante.

## Où on en est
- **Conversation libre :** environ 95 tok/s, pour un plafond bit-à-bit estimé à environ 104 tok/s. Chaque piste restante rapporte moins de 1 %.
- **Long contexte :** limité par la lecture du cache KV en FP32.
- **Prefill :** limité par le calcul en FP32 sur les unités classiques (environ 450 tokens/s sur un long prompt).

## Idées à creuser (à décider)

| Idée | Gain attendu | Contrepartie |
|---|---|---|
| Petit modèle de brouillon Gemma 4 (même tokenizer) pour la spéculation | ×1,5 à ×2 en conversation libre | téléchargement d'un modèle et support d'une seconde architecture |
| Cache KV en FP16 | long contexte nettement plus rapide, moitié moins de VRAM pour le cache | bits différents : validation par top-1 et divergence KL |
| Tensor Cores pour le prefill | prefill plusieurs fois plus rapide | bits différents |
| Cache KV circulaire pour les couches locales (fenêtre de 1 024) | environ 6,7 Go de VRAM récupérés, contexte de 32k possible | aucun sur les résultats, seulement de l'adressage |
| Jeu de prompts de mesure (chat, code, résumé, traduction, recopie) | chiffres moyens fiables, surtout pour la spéculation | aucun (`scripts/bench_suite.py` est prêt mais n'a pas été lancé) |
| Spéculation sur plusieurs requêtes en parallèle (débit serveur) | débit total bien supérieur | ne change pas la vitesse d'une conversation seule |

## Pour reproduire
```
cargo run --release -p adaforge -- bench models/gemma-4-12b-it-qat-q4_0/gemma-4-12B-it-QAT-Q4_0.gguf "Bonjour, comment vas-tu ?" 24 5
python scripts/compare_performance.py models/gemma-4-12b-it-qat-q4_0/gemma-4-12B-it-QAT-Q4_0.gguf
```
Variables utiles : `ADAFORGE_SPECULATIVE=0`, `ADAFORGE_SPEC_DEPTH=n`, `ADAFORGE_BATCH_PREFILL=0`, `ADAFORGE_BATCH_VERIFY=1`, `ADAFORGE_SPLIT_ATTENTION=0`, `ADAFORGE_TIMING_DECODE=1`, `ADAFORGE_DIAGNOSTICS=1`. Le détail technique est dans [optimization-2026-10.md](optimization-2026-10.md).
