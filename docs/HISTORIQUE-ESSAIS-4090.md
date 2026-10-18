# Historique des essais AdaForge / RTX 4090

Ce fichier rassemble les essais réalisés sur AdaForge pour le modèle
`gemma-4-12b-it-qat-q4_0` en GGUF. Il sert de mémoire de projet : ce qui est
en production, ce qui reste expérimental, ce qui a été rejeté et ce qui n'a
pas encore été essayé.

Les débits cités ne sont pas cumulables. Ils dépendent du prompt, de la
longueur générée, des fréquences GPU et de l'activité de la machine. Une
variante n'est retenue que si elle conserve la justesse numérique et présente
un gain suffisamment reproductible dans le protocole disponible.

## Périmètre actuel

- Cible matérielle : une RTX 4090, CUDA, architecture `sm_89`.
- Cible modèle : Gemma 4 12B IT QAT Q4_0, 48 blocs, largeur 3840,
  FFN 15360, vocabulaire 262144.
- Chemin de référence : poids Q4_0/Q6_K, activations et cache KV en FP32,
  calcul GPU, sélection greedy GPU.
- Objectif : dépasser llama.cpp CUDA sur le même modèle, puis approcher
  100 tokens/s en décodage d'une conversation.
- Comparaison fraîche disponible : AdaForge 76,103 tok/s contre llama.cpp
  CUDA 78,390 tok/s, prompt court, 24 tokens, sept répétitions. Le déficit
  observé est 2,286 tok/s sur cette série précise.

## Fondations réalisées et conservées

| Élément | Statut | Ce qui a été validé |
|---|---|---|
| Lecture GGUF Gemma 4 | Conservé | Métadonnées, 667 tenseurs et formats Q4_0/Q6_K inventoriés. |
| CUDA RTX 4090 | Conservé | Diagnostic GPU et kernel FP32 élémentaire. |
| Q4_0 matvec | Conservé | Oracle CPU FP64 et tests de projections réelles. |
| RMSNorm, Q/K/V, Q/K norm, RoPE, V norm | Conservé | Tests unitaires GPU et références CPU. |
| Cache KV, attention causale locale/globale | Conservé | Masque causal, GQA, fenêtres locales et reset contrôlés. |
| Branche attention, FFN et bloc dense | Conservé | Validation couche complète avec poids Gemma. |
| Embeddings Q6_K, tokenizer Gemma et texte | Conservé | Aller-retour texte/tokens et couche depuis les embeddings GPU. |
| 48 blocs, RMSNorm finale, tête Q6_K, softcap | Conservé | 262144 logits contrôlés face au CPU. |
| Génération `ask` et conversation `chat` | Conservé | Réponses complètes, EOS, réutilisation des caches. |
| Poids, buffers, caches et graphes CUDA résidents | Conservé | Pas de recalcul CPU des couches pendant la génération. |
| Greedy sur GPU | Conservé | Retour du seul token sélectionné, pas des 262144 logits. |

## Optimisations retenues dans le chemin par défaut

| Essai | Décision | Preuve disponible |
|---|---|---|
| Conversion matérielle FP16 vers FP32 des échelles Q4/Q6 | Retenue | `cvt.f32.f16`, 63488 encodages FP16 finis vérifiés bit à bit ; ABBA descriptif +3,69 %. Les activations restent FP32. |
| Q4_0 gate/up rangé par tuiles de quatre lignes | Retenue pour 3840 → 15360 | Packing réversible, ordre de réduction FP32 corrigé, pas de duplication VRAM ; ABBA +3,70 % sur le prompt retenu. Désactivation possible : `ADAFORGE_EXPERIMENTAL_TILED_GATE=0`. |
| Projection FFN down, partage K entre quatre warps | Retenue | Le chemin à quatre partitions K est le kernel de production pour la longue projection down. |
| Préparation Q/K/V groupée | Retenue pour têtes 256/512 | 192 lancements supprimés par token ; validations bit à bit et CPU 48 blocs/logits ; ABBA court +2,42 %. Désactivation possible : `ADAFORGE_EXPERIMENTAL_QKV_PREP=0`. |
| QKV groupé / K-V partagé dans les blocs globaux | Retenu | Une projection K/V est calculée une fois lorsque les poids sont partagés ; caches K et V restent distincts. |
| Tête Q6_K et sélection greedy GPU | Retenu | Tête alimentée par le bloc 47 sur GPU, sans transfert des logits complets. |

## Essais réalisés, mais non retenus par défaut

| Famille | Essai | Résultat | Décision |
|---|---|---|---|
| Q4 | Séparation simple des échelles et octets par ligne | Environ 66,24 tok/s contre ~68,1 de référence dans la série concernée | Rejeté. |
| Q4 | Balayage 1/2/8/16 warps par ligne | Aucun gain net | Rejeté sous cette forme. |
| FFN gate/up | Deux lignes voisines par warp | Régression : 64,03 contre 69,28 puis 73,26 tok/s témoins | Rejeté. |
| FFN gate/up | Vecteur d'entrée complet copié en shared | Régression : 63,59 contre 71,64 puis 70,64 | Rejeté. |
| FFN gate/up | Gate et up calculés conjointement avec le mapping natif | Plus lent malgré calcul correct | Rejeté. |
| FFN Q4 | Pipeline `cp.async`, double buffer, 1 ou 4 groupes | Plus lent à chaud et après éviction | Rejeté. |
| FFN down | Disposition tuilée 4 lignes, déroulage 1/2/4/8 | ~20 % plus rapide à chaud, ~5 % plus lente après éviction | Non intégré. |
| FFN gate/up | Déroulage de boucle 1/2/4/8 | Petit gain à chaud, pas de gain à froid | Non intégré. |
| FFN gate/up | Groupes 2/4 lignes par CTA, avec/sans shared | Gain chaud jusqu'à ~20 %, absent ou défavorable à froid | Non intégré. |
| FFN gate/up | Indications de cache PTX `cg/cs/ca/nc` | `cs` favorable en microbenchmark, -0,54 % en ABBA moteur | Option désactivée : `ADAFORGE_EXPERIMENTAL_STREAM_WEIGHTS=1`. |
| FFN | Fusion gate/up + GELU | +0,59 % sur court, ~-2,51 % sur seconde question/64 tokens | Option désactivée : `ADAFORGE_EXPERIMENTAL_FUSED_GELU=1`. |
| Mémoire CUDA | Préférence shared/L1 à 0 % et 100 % | Variations de témoins trop fortes, aucun gain établi | Réglages retirés. |
| FFN | Activations Q8 + `dp4a` | Gain faible, IDs changés, qualité globale non qualifiée | Expérimental, désactivé : `ADAFORGE_EXPERIMENTAL_Q8=1`. |
| Q6_K | Tête/embeddings intercalés sur 4 lignes | ABBA -3,23 %, malgré tests bit à bit | Expérimental, désactivé : `ADAFORGE_EXPERIMENTAL_Q6_TILES=1`. |
| Q6_K | Huit lignes par bloc | ABBA -3,26 % | Désactivé. |
| Q6_K | Chemin brut sans softcap inutile pour greedy | +0,61 %, dérive témoin -3,18 % | Gain non établi, désactivé. |
| Attention | RMSNorm Q/K + RoPE fusionnés | ~+1 % sur deux tests, variations comparables | Expérimental, désactivé : `ADAFORGE_EXPERIMENTAL_NORM_ROPE=1`. |
| Attention | Écriture directe Q/K/V préparés vers cache KV | +0,50 %, dérive témoin -4,04 % | Désactivé : `ADAFORGE_EXPERIMENTAL_PREP_CACHE=1`. |
| Attention | Exponentielles softmax en parallèle | Long +3 %, court -0,92 % | Expérimental : `ADAFORGE_EXPERIMENTAL_ATTENTION_EXP=1`. |
| Projections | K-parallel étendu à QKV ou projection de sortie | Justesse validée, gain moteur non concluant | Expérimental, désactivé. |

## Erreurs ou pistes corrigées pendant les essais

| Sujet | Ce qui s'est passé | Résolution |
|---|---|---|
| Réduction Q4 tuilée initiale | Les canaux étaient additionnés avant les partitions K ; échec au bloc 35 sur validation complète | Ordre corrigé : partitions K d'abord, puis canaux. La tolérance n'a pas été élargie. |
| Premier prototype de tuiles gate/up | Dupliquait les poids en VRAM | Le backend possède désormais soit les matrices natives, soit les poids tuilés ; pas les deux. |
| Hypothèse « poids chauds » | Des microbenchmarks donnaient des gains élevés non vus en génération | Les essais FFN récentes mesurent systématiquement un régime après éviction et un benchmark moteur avant intégration. |
| Estimation de performance | Plusieurs gains isolés semblaient s'additionner | Les gains ne sont plus additionnés ; chaque variante doit être mesurée dans le moteur complet. |

## Méthode de validation utilisée

1. Contrôle local : packing réversible, sorties GPU face à un oracle CPU FP64,
   et comparaison bit à bit quand l'ordre FP32 doit rester identique.
2. Contrôle modèle : selon le changement, couches, 48 blocs et 262144 logits
   comparés avec `atol=rtol=1e-4` et ratio limite 1.
3. Contrôle génération : mêmes IDs de prompt et de sortie que la référence.
4. Mesure : binaire témoin archivé, même modèle/prompt/budget, échauffement,
   KV reset, séries ABBA quand applicable et résultats bruts conservés.
5. Hygiène : compilation release, formatage Rust, Clippy et tests appropriés.

Les relevés `nvidia-smi` avant/après une série sont indicatifs. Ils ne sont
pas des compteurs matériels et ne permettent pas, à eux seuls, d'attribuer la
cause d'un gain ou d'une régression.

## Pistes sérieuses encore ouvertes

| Priorité | Piste | Condition avant adoption |
|---|---|---|
| P0 | Profil decode-only récent d'AdaForge et de llama.cpp | Séparer clairement prefill, CPU, GPU et attentes avant de choisir une cible. |
| P1 | Lectures Q4 vectorisées réellement alignées et dispatch par formes Gemma | Conserver Q4_0 groupe 32, FP32 et une validation indépendante. |
| P2 | Optimiser la tête Q6_K et fusionner tête + top-k/greedy dans le graphe | Mesurer la part réelle des lancements et du retour hôte. |
| P2 | W4A16/Tensor Cores, spécialisé sm_89 | Nouveau chemin de précision : oracle propre, écarts logits/perplexité/réponses avant activation. |
| P2 | Prefill par lots via GEMM et attention par blocs | Cible le temps du premier token et prépare la vérification spéculative. |
| P3 | Boucle de génération GPU, propositions n-gram/MTP et vérification parallèle | Dépend d'un prefill par lots fiable et d'un taux d'acceptation mesuré. |
| P3 | KV FP16/INT8/4-bit, cache circulaire local, attention long contexte | Utile principalement quand le contexte long devient le goulot. |

## Documents et données sources

- Plan vivant : [optimization-plan.md](optimization-plan.md)
- Tuiles Q4 et ABBA : [optimization-tiled-q4.md](optimization-tiled-q4.md), [optimization-async-and-abba.md](optimization-async-and-abba.md)
- Conversion FP16 et comparaison llama.cpp : [optimization-hardware-half.md](optimization-hardware-half.md)
- Préparation QKV : [optimization-qkv-prep.md](optimization-qkv-prep.md)
- Attention : [optimization-norm-rope.md](optimization-norm-rope.md), [optimization-prep-cache.md](optimization-prep-cache.md), [optimization-attention-exp.md](optimization-attention-exp.md)
- FFN : [optimization-gate-reuse.md](optimization-gate-reuse.md), [optimization-ffn-cold-probes.md](optimization-ffn-cold-probes.md), [optimization-cache-stream.md](optimization-cache-stream.md), [optimization-fused-gelu.md](optimization-fused-gelu.md)
- Q6 : [optimization-q6-launch.md](optimization-q6-launch.md), [optimization-q6-raw.md](optimization-q6-raw.md)
- Résultats bruts : dossier `benchmarks/`.

