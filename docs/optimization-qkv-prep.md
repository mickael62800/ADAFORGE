# Préparation Q/K/V groupée — 18 octobre 2026

## Hypothèse et implémentation

La préparation originale lance trois normalisations et deux rotations RoPE par couche. La variante précédente ne fusionnait que chaque norme Q/K avec sa rotation : trois lancements restaient nécessaires. Cette expérience regroupe la normalisation Q, K, V et les rotations Q/K dans un seul lancement, avec un bloc CUDA par tête. Les blocs Q/K effectuent RMSNorm puis RoPE ; les blocs V normalisent sans rotation.

Les opérations et l'ordre de réduction FP32 restent inchangés. Un tampon FP32 en mémoire partagée matérialise les valeurs normalisées avant la rotation. Les fréquences et la trigonométrie FP64 existantes restent utilisées. Aucun changement du format des poids ou des caches KV FP32.

192 lancements sont supprimés par token pour 48 couches par rapport au chemin de production séparé. Les écritures du cache KV et l'attention sont encore des lancements distincts. Les couches globales lisent le même tampon de projection pour K et V mais écrivent deux résultats distincts : K est tourné, V ne l'est pas.

## Activation

`ADAFORGE_EXPERIMENTAL_QKV_PREP=1` active la variante pour les têtes de 256 ou 512 canaux ; elle est activée par défaut depuis sa qualification ; `0` rétablit le chemin séparé. Ce réglage prend la priorité sur la fusion Q/K seule. `ADAFORGE_DIAGNOSTICS=1` rapporte `grouped_qkv_prep_layers` dans `ADAFORGE_CONFIG`.

Référence archivée avant changement : `tools/optimization-qkv-prep/baseline.exe`.

## Vérifications ciblées

Comparaisons bit pour bit à largeurs 256/512, échelles signées, plusieurs têtes et positions 0, 1, 127, 4095, 131071, 262143. Chemins statique et dynamique testés. Les entrées groupées peuvent être aliasées en lecture. Q/K sont comparés aux kernels séparés ; V est comparé à la norme seule afin de détecter toute rotation indue.

Les tests de blocs résidents, session et graphe passent avec le réglage expérimental ; le test de graphe utilise effectivement des têtes de largeur 256 et compare des tokens/positions variables ainsi que les resets. Compilation release, formatage, clippy sans avertissement et tests de comptabilité du benchmark réussis.

## Vérification du modèle complet

48 blocs et 262144 logits contrôlés face au CPU indépendant : erreur norme finale 0,00008223 (ratio 0,29132), logits bruts 0,00005980 (ratio 0,42836), logits après softcap 0,00005519 (ratio 0,42835). Toutes les limites restent à 1, atol=rtol=1e-4. Sortie conservée dans `benchmarks/optimization-qkv-prep-logits.txt`.

## Comparaisons non profilées et décision

ABBA, cinq mesures conservées par processus, un échauffement interne plus une requête mesurée écartée, poids et capacités identiques, identifiants de sortie identiques sur toutes les séries.

| Test | Référence avant/après | Variante, deux séries | Moyenne des médianes : référence → variante | Delta descriptif | Dérive des contrôles |
|---|---|---|---|---|---|
| Bonjour, 24 tokens | 74.209 / 74.690 | 75.480 / 77.021 | 74.449 → 76.251 | 2.419 % | 0.648 % |
| Comptage, 64 tokens | 65.848 / 69.232 | 70.780 / 71.883 | 67.540 → 71.332 | 5.614 % | 5.140 % |

Données : `benchmarks/optimization-qkv-prep-abba.json` et `benchmarks/optimization-qkv-prep-counting-abba.json`. Chaque médiane candidate dépasse les deux contrôles de son test. Le contrôle du test long dérive sensiblement : son delta ne constitue pas une estimation précise du gain. Le test court indique environ +2,42 %, soit +1,80 token/s dans cette session. Aucun débit général de 80 ou 100 tokens/s n’est établi.

La variante devient le défaut pour les largeurs qualifiées 256/512. Le chemin séparé reste disponible par `ADAFORGE_EXPERIMENTAL_QKV_PREP=0`. Décision fondée sur les calculs conservés, la vérification CPU complète et les deux comparaisons positives, avec une portée limitée aux cas testés. Les autres expériences Q6/QKV ne sont pas activées. Prochaine cible : attention ou fusion de l’écriture cache après mesure dédiée.
