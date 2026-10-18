# Kernels Ada

`vector_add.cu` : addition FP32 de diagnostic, avec contrôle des limites. Source embarqué dans le backend Rust et compilé par NVRTC pour `compute_89`, chargé sur la RTX 4090 sm_89. Aucun kernel d’inférence à ce stade.

`q4_matvec.cu` : produit matrice Q4_0 × vecteur FP32, accumulation FP32. Un warp par ligne, quatre par bloc. Poids déquantifiés à la volée. Diagnostic de justesse sur poids réels ; aucune inférence complète pour le moment.

`rms_norm.cu` : RMSNorm FP32 à échelle directe, réduction partagée et un bloc de 256 threads par vecteur. Support des largeurs non multiples de 256.

`rope.cu` : rotation NeoX par paires de moitiés de tête. Fréquences et angles FP64 pour cette version de justesse, sorties FP32. Paramètres locaux et facteurs globaux préparés depuis le modèle.

`rms_norm.cu` expose aussi `rms_norm_unscaled` pour V, sans argument poids. La réduction commune est spécialisée par template ; seul le mode pondéré accède aux échelles.

`attention.cu` : premier decode attention causal FP32, un bloc par tête Q, GQA, fenêtre optionnelle et softmax stable. Scratch des scores en mémoire partagée ; limite backend 4096 tokens. Version de justesse à optimiser ultérieurement.

`ffn.cu` : activation GELU(tanh) de gate multipliée par up, puis addition résiduelle avec échelle finale de couche. Les projections et normes réutilisent les kernels existants.

`embedding_q6.cu` : déquantification Q6_K des lignes sélectionnées, un thread par valeur, avec multiplicateur FP32 optionnel. La table complète n’est pas transférée dans ce diagnostic.
