# Politique de cache des poids FFN — 18 octobre 2026

Quatre politiques PTX comparées sur le rangement entrelacé existant : cg, cs, ca, nc. Même ordre FP32. Matrices gate/up concaténées des blocs 0 et 5, trois entrées, trois tailles synthétiques ; packing inverse, égalité bit à bit et oracle CPU réussis.

La politique streaming cs passe d'environ 123–124 µs à 115 µs après éviction de 128 Mio. Les autres politiques ne présentent pas de gain net. Ce microbenchmark ne représente pas à lui seul un flux de génération.

Intégration optionnelle : ADAFORGE_EXPERIMENTAL_STREAM_WEIGHTS=1, désactivée par défaut. Modifie uniquement les lectures des codes/échelles du kernel entrelacé, sans changer les formats ou dupliquer les poids.

Qualification moteur ABBA : 5 requêtes retenues par série, une supplémentaire écartée, 24 tokens, prompt Bonjour identique. Témoins : 75,9698 et 76,0504 tokens/s ; candidats : 75,4799 et 75,7142. Variation descriptive -0,54 %, dérive des témoins +0,11 %. Tous les IDs du prompt et de génération sont identiques. Aucun gain moteur démontré : option maintenue désactivée.

Traces : benchmarks/ffn-cache-probe.txt et benchmarks/optimization-cache-stream.json. Le binaire témoin est archivé dans tools/optimization-cache-stream/baseline.exe. Le test GPU FFN avec oracle CPU est exécuté avec le chemin entrelacé et streaming forcés ; Clippy des exemples passe.

Conclusion : le gain du microbenchmark à froid ne se transpose pas au moteur. Ne pas poursuivre cette politique seule. La prochaine investigation devrait porter sur un profil récent de génération et des changements structurels (fusion gate/up/activation ou tête dans le graphe), plutôt que d'extrapoler les microbenchmarks.
