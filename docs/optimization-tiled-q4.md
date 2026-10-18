# Q4_0 rangé par tuiles de quatre lignes

État actuel : les [mesures ABBA suivantes](optimization-async-and-abba.md) ont permis de retenir le kernel synchrone par défaut pour le FFN 3840/15360. Les premières séries ci-dessous documentent sa qualification progressive.

Le prototype est dans `kernels/ada/q4_tiled_probe.cu`, avec un banc autonome `crates/cuda-backend/examples/tiled_probe.rs`. Il conserve tous les nibbles et les bits des échelles FP16 d'origine. Le banc vérifie une reconstruction inverse octet pour octet avant le transfert GPU.

Le rangement intercale quatre lignes, puis les mots de 16 bits d'un bloc de 32 poids. Un warp lit ainsi quatre lignes avec huit lanes par ligne. Deux kernels sont comparés : une boucle complète sur K par warp, puis un partage de K entre quatre warps du CTA. Le second réduit les quatre partitions avant les huit lanes de canaux, dans l'ordre FP32 du `q4_row` natif. Le kernel FFN réunit gate et up dans une seule allocation logique de lignes et un seul lancement.

## Méthode

Le banc compare la projection Q de la couche 0, gate et down, sur zéro, signal structuré et pseudo-aléatoire, avec oracle CPU FP64 et tolérance initiale `atol=rtol=1e-4`. Les ordres des variantes alternent ; vingt mesures CUDA après deux échauffements sont utilisées. Deux régimes : répétition directe, puis une passe sur un buffer de 128 Mio avant chaque kernel hors chronométrage. Cette passe vise à évincer le L2 mais n'est pas une mesure matérielle de son contenu ou de la bande passante.

Le banc vérifie aussi l'égalité bit à bit de la variante split avec le kernel natif une ligne par warp pour Q et gate. Down utilise déjà une autre réduction K-parallel et reste comparé par tolérance à l'oracle.

La première réduction testée effectuait les canaux avant les partitions K. Malgré un microbenchmark favorable et 27 tests GPU passants, elle a échoué au contrôle réel des 48 couches au bloc 35, position 5 : erreur absolue 0,00013858 pour une référence -0,35079038. **Cette variante est rejetée.** Son rapport `benchmarks/optimization-tiled-gate.json` (73,45 contre 70,97/71,25 tok/s) ne constitue pas un gain valide. La tolérance n'a pas été élargie ; l'ordre de réduction a été corrigé.

## Chemin expérimental

La version corrigée passe le contrôle des 48 couches et des 262144 logits : ratio maximal 0,29132 pour la norme finale, 0,42836 pour les logits bruts et 0,42835 après softcap, pour une limite de 1. Ces résultats retrouvent ceux du chemin FP32 de référence ; aucune tolérance n'a changé.

| Prompt / longueur de sortie demandée | Avant | Tuiles corrigées | Avant répété |
|---|---:|---:|---:|
| Bonjour / 24 tokens | 71,84 | 72,55 | 70,61 |
| Explication GPU / 48 tokens | 68,34 | 72,76 | 70,27 |

Débits médians sur cinq répétitions par processus, après un échauffement. Les IDs sont identiques sur chaque série avant/après/avant. Le gain face au contrôle le plus rapide est respectivement d'environ 1,0 % et 3,5 %. L'amplitude et la dérive des contrôles imposent de rester prudent : ce sont deux prompts, pas un benchmark général ni une preuve que 80 tok/s sont atteints. Rapports : `benchmarks/optimization-tiled-gate-exact.json` et `benchmarks/optimization-tiled-gate-explanation.json`.

Les 27 tests GPU passent avec le drapeau tiled actif après correction. Le banc autonome couvre aussi les lignes partielles 1/5/9 avec 96 colonnes, et vérifie l'identité bit à bit Q/gate face au kernel natif. Les mesures et contrôles du banc sont conservés dans `benchmarks/tiled-kernel-probe.json`.

`verify_generation.py` passe également avec le drapeau actif et la référence officielle llama.cpp CUDA b11669 : réponse complète identique, arrêt de modèle, préfixe à deux tokens et limites invalides vérifiés. Formatage Rust et clippy avec avertissements interdits passent.

Le packing gate/up est maintenant actif par défaut pour width=3840 et intermediate=15360. L'override historique `ADAFORGE_EXPERIMENTAL_TILED_GATE=0` sélectionne le kernel natif ; `1` force les tuiles. Le chemin ne s'applique pas au prototype Q8. Les poids Q4_0 et les activations FP32 restent inchangés en précision. Le temps de packing et le chargement sont exclus du débit de décodage.

Le premier prototype doublait le stockage gate/up. Cette duplication est désormais supprimée : le backend possède soit les deux matrices natives, soit les codes et échelles tuilés, ainsi que les poids down communs aux deux chemins. Pour Gemma, la version tuilée occupe exactement le même nombre d'octets de poids gate/up que la version native : 66 355 200 par couche. Les 3 185 049 600 octets supplémentaires (2,97 Gio) du prototype précédent ne sont plus alloués. Ce chiffre concerne les buffers de poids demandés à CUDA, pas une mesure de la VRAM totale du processus ou de la carte.

Un test de régression vérifie ce budget dans les chemins natif et tuilé en plus de l'oracle FFN. Les 27 tests GPU passent après la modification ; le test FFN avec le drapeau actif passe aussi.

Troisième comparaison, après suppression des duplications : prompt de fonction Rust, sortie demandée de 96 tokens, cinq répétitions après échauffement. Avant 62,96, tuiles 65,53, contrôle répété 64,98 tok/s ; IDs identiques. Rapport : `benchmarks/optimization-tiled-gate-no-duplicate-code.json`. L'écart est de 0,56 tok/s face au contrôle le plus rapide, avec une dérive de 2,02 tok/s entre les contrôles. On ne présente donc pas 2,57 tok/s comme un gain acquis. Les deux premières lignes du tableau ci-dessus concernaient encore le prototype avec duplication.

La validation numérique et la suppression des duplications sont acquises. Après les trois premières séries, une qualification ABBA supplémentaire a montré des contrôles stables et un gain descriptif de 3,7 %, retenu pour cette géométrie. Les gains du microbenchmark ne sont pas assimilés au débit de génération et la généralisation à d'autres contextes reste à mesurer.

Il n'existe pas de seuil minimal fixe en tokens/s : +2 à +3 tok/s autour de 72 représente déjà environ 3 à 4 %, et +1 peut être utile pour un changement simple et stable. La décision dépend de la répétabilité, de la précision, des régressions sur d'autres formes/contextes et du coût en mémoire ou au démarrage. Les petits gains sont acceptables ; un effet non distinguable de la variation des contrôles reste une hypothèse.

Commande du banc autonome :

```powershell
cargo run --release -p adaforge-cuda-backend --example tiled_probe -- models/gemma-4-12b-it-qat-q4_0/gemma-4-12B-it-QAT-Q4_0.gguf
```

Le script `compare_optimization.py` accepte `--prompt`, `--tokens`, `--runs`, `--abba` et `--discard-leading` pour élargir les comparaisons. Il refuse le drapeau Q8 et conserve les chemins des binaires ainsi que la valeur de l'override tiled dans les nouveaux rapports.
