# Premier essai après la revue des moteurs externes

Deux variantes du partage L1/shared ont été testées sur les seules fonctions `q4_gate_up` et `q4_matvec_kparallel` du FFN. Les variantes fixaient `CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT` respectivement à 0 et 100 via `CudaFunction::set_attribute`, une fois au chargement. Tous les autres kernels gardaient le choix CUDA automatique. Aucun changement des poids ou de la précision.

| Préférence shared | Avant | Variante | Avant répété | Conclusion |
|---|---:|---:|---:|---|
| 0 % | 68,98 | 70,40 | 73,91 | Dérive trop importante pour conclure à un gain |
| 100 % | 69,71 | 69,86 | 71,54 | Aucun avantage démontré |

Débits en tokens/s, médianes de cinq répétitions par processus après un échauffement, 24 tokens de sortie, même prompt Bonjour. Tous les IDs de prompt et de sortie sont identiques entre les trois séries de chaque essai. Les fichiers bruts sont `benchmarks/optimization-carveout-zero.json` et `benchmarks/optimization-carveout-full.json`.

Le réglage expérimental a été retiré après ces mesures. Le binaire antérieur est conservé sous `tools/optimization-carveout-baseline/adaforge.exe`. Ces essais ne couvrent ni les autres familles de kernels, ni un nouveau pipeline utilisant beaucoup de shared memory : M03 reste une piste partiellement explorée.

Un profil récent a ensuite été capturé sans préférence explicite avec Nsight Systems et `--cuda-graph-trace=node`. Son résumé est dans `benchmarks/profile-current-kparallel-summary.json`. Le Q4 représente 73,1 % du temps cumulé des kernels de cette trace ; gate/up seul 38,3 %, down 17,6 %, QKV 10,8 %, projection attention 6,4 %. Le prefill et le decode sont mélangés et des pointes de latence sont présentes ; ces pourcentages ne constituent ni un profil decode pur ni une mesure de bande passante DRAM.

L'ordre pratique devient donc :

1. Mesures multi-prompts et profils decode seul des deux moteurs (M01, M02, M30).
2. Gate/up Q4 : réutilisation des activations et dispatch par forme (M04, M05, M26), puis tuiles Q4_0 groupe 32 et chargements alignés (M06, M07, M27).
3. Down, QKV et tête Q6, selon les gains restant après ce premier travail.
4. Prefill par lots, puis vérification spéculative et MTP.
5. KV réduit et attention longue : à prioriser lorsque le contexte devient le goulot.

Aucun nouveau gain de vitesse n'est revendiqué pour cette passe.
