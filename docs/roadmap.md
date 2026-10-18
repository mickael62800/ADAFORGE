# Étapes

1. Base : workspace, frontières et compilation — présent dans ce squelette.
2. Fixer un premier modèle, son format de poids et le protocole de comparaison sur 4090.
3. Intégrer CUDA : détection du GPU, buffers, premier kernel et vérification numérique — implémenté et validé sur RTX 4090.
4. Lire et valider les métadonnées et poids du modèle cible — inventaire GGUF et validation des plages implémentés ; lecture numérique des poids à venir.
5. Implémenter les opérations nécessaires et une passe de référence correcte.
6. Générer des tokens : tokenisation, prefill, cache KV, decode et sampling.
7. Profiler et optimiser sur RTX 4090, avec contrôles de qualité.
8. Ajouter le serveur, le streaming et les usages concurrents selon les besoins.

Chaque étape devra produire un résultat vérifiable avant d’élargir le périmètre.

Premier jalon de l’étape 5 validé : matvec Q4_0 sur deux matrices réelles de Gemma (attention Q et FFN down), comparaison CPU FP64 et tests GPU synthétiques. Les opérations F32/Q6_K, les autres opérateurs et la passe complète restent à implémenter.

RMSNorm validée sur les poids Gemma (largeurs 3840 et 256) et sur tailles synthétiques, avec lecture F32 et epsilon depuis GGUF. Enchaînement RMSNorm + projection Q réalisé sur le premier bloc, avec buffers GPU réutilisés.

Chaîne RMSNorm → Q validée sur Gemma, quatre vecteurs et toutes les sorties ; test GPU synthétique avec cinq lignes pour couvrir un dernier groupe de warps incomplet. Les projections Q/K/V brutes sont validées sur les couches 0 et 5 ; normalisation des têtes, RoPE et attention restent à développer.

Normalisation par tête Q/K ajoutée à la chaîne GPU et validée sur les couches 0 et 5, plus tests synthétiques multi-têtes. V brut est préservé. RoPE, RMSNorm V sans échelle et attention restent à implémenter.

RoPE NeoX ajoutée après normalisation Q/K, avec sélection locale/globale, facteurs proportionnels GGUF et vérification CPU/GPU jusqu’à la dernière position du contexte. Prochaines briques : RMSNorm V sans échelle, cache KV et attention causale.

RMSNorm V sans échelle ajoutée à la chaîne et validée sur les couches 0/5 et sur données synthétiques multi-têtes, sans RoPE sur V. La préparation Q/K/V est vérifiée ; prochaines étapes : cache KV puis attention causale.

Cache KV contigu FP32 implémenté et testé indépendamment : ajout par lots GPU/GPU, préfixe conservé, capacité protégée, reset et réutilisation. Géométries locale/globale lues dans GGUF. Étape suivante : brancher K tourné et V normalisé sur ce cache, puis calculer l’attention.

Chaîne positionnelle connectée au cache GPU : huit positions 0–7, K après RoPE et V après normalisation ajoutés sans retour hôte intermédiaire. Préfixes comparés au CPU et conservation exacte des anciens éléments vérifiée. Validation sur blocs local/global. Prochaine brique : attention causale utilisant ce cache.

Attention causale GQA depuis le cache ajoutée à la chaîne : échelle Gemma 4 = 1, fenêtre locale 1024 ou globale, softmax stable. Vérification des positions 0–7 et tests GPU synthétiques de causalité/fenêtre 1025 tokens. Version de référence limitée à 4096 tokens. Étape suivante : projection de sortie d’attention et normalisation/résiduel, puis FFN.

Branche d’attention complétée : projection Q4_0 de sortie, RMSNorm post-attention et résiduel sur GPU. Validée sur les blocs 0 et 5, huit positions, plus test isolé avec dimension 257. Prochaine brique : FFN (norme d’entrée, projections gate/up, activation Gemma, projection down, norme et résiduel).

FFN dense ajouté à la chaîne : RMSNorm, gate/up Q4_0, GELU(tanh) multipliée par up, down Q4_0, post-norme, résiduel et échelle de couche. Validation d’un bloc dense sur les couches locale/globale et test synthétique complet. Restent : embeddings/tokenisation, lecture Q6_K pour embeddings, enchaînement des 48 blocs, norme et projection finales, sampling et validation de bout en bout avant optimisation.

Embeddings Q6_K par IDs numériques ajoutés : lecture sélective GGUF, déquantification CPU/GPU, échelle sqrt(hidden_size), ordre/répétitions et limites du vocabulaire vérifiés. Étapes suivantes : connecter l’embedding à la première couche, tokenisation Gemma et enchaînement des blocs.

Embeddings réels connectés au premier bloc dense sur GPU, sans réinjection des résultats CPU. Huit IDs numériques et toutes les sorties du bloc 0 vérifiés. Les blocs suivants ne sont pas encore enchaînés. Prochaine brique : tokenisation du texte Gemma et validation de ses IDs.
