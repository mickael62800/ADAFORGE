# Tête Q6 : suppression du softcap inutilisé en greedy

Essai du 18 octobre 2026, guidé par la durée de la projection finale dans les captures Nsight.

## Modification

Le chemin de génération sélectionne déjà le maximum des logits bruts. Pourtant la projection écrivait aussi tous les logits après softcap, sans lecture de ce tampon par la sélection. Une spécialisation CUDA `q6_matvec_raw` retire ces calculs et écritures, en conservant exactement la déquantification Q6_K, les opérations et les réductions FP32 des logits bruts.

Activation expérimentale : `ADAFORGE_EXPERIMENTAL_Q6_RAW=1`. Réglage désactivé par défaut. Le diagnostic `logits-check` garde sa projection complète et sa vérification du softcap. Ce changement ne définit pas un chemin de sampling probabiliste : il concerne la sélection greedy existante.

## Validation

- Test GPU avec poids signés, plusieurs blocs et groupe de lignes incomplet : oracle CPU réussi pour la tête résidente expérimentale.
- Test dédié comparant les kernels complet et brut : logits bruts identiques bit pour bit ; tampon softcap sentinelle intact avec la spécialisation brute.
- Compilation release et clippy sans avertissement réussis.
- Comparaison ABBA : identifiants de prompt et de sortie identiques sur toutes les séries.

## Mesures sans profiler

24 tokens, cinq mesures conservées par processus, un échauffement interne et une requête supplémentaire écartée.

| Série | Médiane tokens/s |
|---|---:|
| Référence avant | 75,954 |
| Variante | 75,059 |
| Variante répétée | 75,352 |
| Référence après | 73,537 |

Moyenne des médianes : référence 74,746 ; variante 75,205, soit +0,614 %. Les contrôles dérivent de -3,182 %. Le résultat ne démontre pas un gain reproductible ; la variante reste désactivée. Aucun gain adopté pour cet essai.

Données brutes : `benchmarks/optimization-q6-raw-abba.json`. Référence archivée : `tools/optimization-q6-raw/baseline.exe`.

Cette suppression seule ne résout pas le coût de la tête. Elle pourra être réévaluée dans une fusion projection/sélection. Les prochaines expériences devraient viser le rangement/déquantification Q6 ou la fusion RMSNorm/RoPE identifiée dans les traces, plutôt que supposer un gain lié au nombre d'opérations supprimées.
