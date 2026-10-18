# Prototype Q8 / dp4a pour RTX 4090

Ce chemin est expérimental, désactivé par défaut. Le chemin validé conserve ses activations FP32 et ses tolérances. Les poids restent Q4_0. Le prototype quantifie les activations des trois projections du FFN en INT8 symétrique par groupes de 32, avec une échelle FP32, et accumule les produits entiers avec `dp4a` avant remise à l’échelle FP32. L’attention, les normes, GELU, les résiduels, le cache KV et la tête Q6_K gardent leur précision actuelle.

## Ce qui est vérifié

- Quantification : maximum absolu par groupe, division par 127, arrondi au pair, plage [-127,127], groupe nul et propagation des entrées non finies par une échelle NaN. Un groupe non fini ne doit pas être silencieusement rendu valide.
- Produit Q4/Q8 : référence CPU utilisant les activations quantifiées, avec frontières FP32 et accumulation FP64 ; la tolérance de calcul reste `atol=rtol=1e-4`. L’écart à la référence FP32 non quantifiée est mesuré séparément, sans élargir le seuil pour le masquer.
- FFN synthétique complet : norme d’entrée, quantification, gate/up, GELU(tanh), seconde quantification, down, norme de sortie et résiduel, face à un oracle CPU indépendant avec activations quantifiées.
- Le probe utilise les poids réels de la couche 0 (attention Q, gate et down), quatre familles de vecteurs synthétiques, un échauffement et 20 mesures par combinaison. Chaque mesure Q8 inclut aussi sa quantification. Ce test à poids réutilisés ne représente pas la lecture des 48 couches en génération complète.

Sur les vecteurs structurés et pseudo-aléatoires testés, l’écart L2 relatif de projection à FP32 est d’environ 0.3–0.4 %. Cela ne mesure ni l’erreur cumulée du modèle ni sa qualité générale. En particulier, la réponse au message de salutation change : le prototype produit 27 tokens et omet « de demander », là où le chemin validé produit 29 tokens. Une comparaison de vitesse n’exige donc pas des IDs identiques entre ces deux précisions ; elle exige le même prompt et le même nombre de tokens, et rapporte explicitement la divergence.

## Reproduction

```powershell
cargo run --release -p adaforge-cuda-backend --example q8_probe -- models/gemma-4-12b-it-qat-q4_0/gemma-4-12B-it-QAT-Q4_0.gguf
cargo test -p adaforge-cuda-backend experimental_q8_ffn -- --ignored
cargo test -p adaforge-cuda-backend --example q8_probe -- --ignored
python scripts/compare_q8.py models/gemma-4-12b-it-qat-q4_0/gemma-4-12B-it-QAT-Q4_0.gguf
```

Le script compare successivement FP32 / Q8 / FP32, cinq répétitions après un échauffement par processus, 24 tokens de sortie et caches remis à zéro. Il isole la variable d’environnement dans ses sous-processus, vérifie la stabilité des IDs dans chaque précision et l’égalité des deux passages FP32, puis rapporte l’égalité ou la différence entre les précisions. Le rapport est `benchmarks/q8-experimental.json`. Le probe est conservé dans `benchmarks/q8-kernel-probe.json`.

Pour un essai manuel uniquement :

```powershell
$env:ADAFORGE_EXPERIMENTAL_Q8='1'
try {
    cargo run --release -p adaforge -- ask models/gemma-4-12b-it-qat-q4_0/gemma-4-12B-it-QAT-Q4_0.gguf "Bonjour, comment vas-tu ?" 64
} finally {
    Remove-Item Env:ADAFORGE_EXPERIMENTAL_Q8
}
```

La variable doit être retirée avant les diagnostics FP32 (`layer-check`, `logits-check`, etc.) : ces commandes conservent leur oracle strict et ne valident pas un nouveau choix de précision. `compare_performance.py` refuse le mode expérimental pour éviter de confondre la comparaison CUDA validée et ce prototype. Aucun résultat à 100 tok/s ni équivalence générale de qualité n’est revendiqué.

## Première mesure complète

Avec la projection down Q8 répartie entre quatre warps : 70.90 tok/s en FP32 avant, 73.37 en Q8 et 71.34 en FP32 après. Le rapport est conservé sous `benchmarks/q8-experimental-kparallel.json`. Ce gain d’environ 3 % ne justifie pas une activation par défaut avec une qualité encore non évaluée. Un second essai utilise une ligne par warp pour down.

Le profil `profile-q8-experimental.nsys-rep` comprend échauffement et prefill, avec traçage des nœuds du graphe. Il attribue 35.0 % du temps des kernels au gate/up Q8, 17.2 % au down Q8 et 0.9 % à la quantification. Ces pourcentages ne sont pas un profil de décodage seul et ne donnent pas un débit de génération.


Avec down Q8 à une ligne par warp : 73.69 tok/s en FP32 avant, 74.71 en Q8 et 73.21 en FP32 après (`benchmarks/q8-experimental.json`). L’écart est faible et les deux séries ne démontrent pas un grand gain reproductible. Le chemin Q8 reste donc expérimental. Son format d’activation est interne (32 INT8 + échelle FP32 séparée), pas un nouveau format de poids GGUF ni une lecture Q8_1.

Les tests du workspace, Clippy, les 26 tests GPU du backend et le test GPU de quantification de l’exemple passent. La qualité générale du modèle n’est pas évaluée par ces tests de calcul.


La vérification fonctionnelle `python scripts/verify_q8.py <modele.gguf>` passe sur le transcript testé : mémorisation du prénom, reset et réponses identiques entre graphe CUDA et chemin direct. Résultats dans `benchmarks/q8-functional.json`. Après retour au mode normal, `verify_generation.py` confirme la réponse exacte de référence, l’arrêt du modèle et les limites de génération. Ces vérifications fonctionnelles ne constituent pas une évaluation de qualité sur un corpus.
