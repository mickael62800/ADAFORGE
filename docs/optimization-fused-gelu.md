# Fusion gate/up et GELU — 18 octobre 2026

Prototype optionnel ADAFORGE_EXPERIMENTAL_FUSED_GELU=1, désactivé par défaut. Un CTA de 256 threads calcule quatre lignes gate et quatre lignes up correspondantes, puis applique GELU(tanh) et le produit. La réduction FP32 de chaque projection et l'expression GELU sont conservées ; la communication passe par shared memory. Le packing existant est réutilisé sans duplication des poids. Le chemin est limité aux matrices entrelacées dont le nombre de lignes est multiple de quatre.

Le chemin fusionné supprime un lancement GELU par couche. Les buffers gate/up restent alloués pour conserver un prototype simple, mais ne sont pas écrits/lus sur ce chemin. Cela ne réduit pas la lecture des poids.

Test GPU FFN contre référence CPU : réussi, chemin entrelacé et fusion forcés, entrées nulles/non nulles et résiduel. Clippy all-targets et formatage réussis. Benchmark ABBA de génération : voir benchmarks/optimization-fused-gelu.json. Binaire témoin archivé dans tools/optimization-fused-gelu/baseline.exe.

## Mesures et décision

- optimization-fused-gelu : témoins [74.71661288153386, 74.5674923341376], candidats [74.95045123430357, 75.21577935483828], variation descriptive 0.591 %, dérive des témoins -0.200 %.
- optimization-fused-gelu-long : témoins [70.92141895522393, 70.60533879467496], candidats [67.75477975018275, 70.22295006390289], variation descriptive -2.508 %, dérive des témoins -0.446 %.

Premier protocole : Bonjour, 24 tokens, 5 mesures retenues par série. Second : explication du ciel bleu, budget 64 tokens, 3 mesures retenues par série. Une requête supplémentaire écartée dans chaque série ; ordre ABBA. IDs strictement identiques dans chaque comparaison. Les séries ne sont pas des intervalles de confiance.

Couche réelle 0 : 30720 sorties contrôlées face au CPU, erreur maximale 0,00000572, ratio de tolérance 0,02099. Aucun assouplissement de tolérance.

Décision : ne pas activer par défaut. Le gain court +0,59 % ne se reproduit pas sur le deuxième prompt (environ -2,51 %). Le prototype reste disponible pour investigation ; aucune accélération générale revendiquée. La suppression de 48 lancements ne suffit pas à établir un gain : elle modifie aussi l'organisation des blocs et la communication gate/up. Il faut profiler ces coûts avant une nouvelle variante.
