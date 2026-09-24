# Related Work Notes

Access date: 2026-09-15.

## Dataset Sources

- AMIGOS official dataset page: https://eecs.qmul.ac.uk/mmv/datasets/amigos/
- DEAP original paper: Koelstra et al., "DEAP: A Database for Emotion Analysis Using Physiological Signals", IEEE Transactions on Affective Computing, 2012, https://doi.org/10.1109/T-AFFC.2011.15
- MAHNOB-HCI database page: https://mahnob-db.eu/
- MAHNOB-HCI paper page: Soleymani et al., "A Multimodal Database for Affect Recognition and Implicit Tagging", https://research.utwente.nl/en/publications/a-multimodal-database-for-affect-recognition-and-implicit-tagging

## Method Sources

- Domain-adversarial training and gradient reversal: Ganin et al., "Domain-Adversarial Training of Neural Networks", JMLR 2016, https://www.jmlr.org/papers/v17/15-239.html

## Notes

- EA-FUSION is treated as related work, not as an architecture to reproduce in this repository.
- Subject-adversarial training can hurt emotion prediction if subject-specific structure carries useful label signal or optimization becomes unstable.
- Low subject-classifier accuracy and embedding plots are diagnostics, not proof of subject invariance.
- Statistical comparisons must respect participant clustering; windows are not independent people.

