# Hand Hybrid Parameter Tuning

- Signs: `abdomen, hello, love, accept, across, airplane, alphabet, angry, animal, answer`
- Configurations: `9`
- Total runs: `90`
- Runtime: `155.05 seconds`

The balanced score prioritizes low wrist-relative shape jerk, then topology, while penalizing lost motion, lost coverage, joint violations, and bone-length variation.

| Rank | Configuration | Coverage | Bone span ↓ | Max bend ↓ | Joint violations ↓ | Topology ↓ | Motion ↑ | Shape jerk ↓ | Score ↓ |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | smooth_s12_r2_a115 | 58.17% | 0.00% | 115.00° | 0.00% | 27.04% | 98.93% | 0.024665 | 0.5331 |
| 2 | smooth_s12_r2_a110 | 58.17% | 0.00% | 110.00° | 0.00% | 28.10% | 98.93% | 0.024802 | 0.5428 |
| 3 | smooth_s10_r2_a115 | 58.17% | 0.00% | 115.92° | 0.00% | 27.46% | 99.00% | 0.029183 | 0.5979 |
| 4 | balanced_s10_r2_a110 | 58.17% | 0.00% | 110.88° | 0.00% | 27.54% | 98.93% | 0.029314 | 0.5997 |
| 5 | smooth_s10_r1_a115 | 58.17% | 0.00% | 115.47° | 0.01% | 27.25% | 97.65% | 0.036611 | 0.6975 |
| 6 | default_s08_r1_a115 | 58.17% | 0.00% | 116.00° | 0.01% | 27.18% | 97.44% | 0.040595 | 0.7509 |
| 7 | joint_s08_r1_a110 | 58.17% | 0.00% | 110.96° | 0.00% | 27.61% | 97.36% | 0.040689 | 0.7629 |
| 8 | joint_s08_r1_a105 | 58.17% | 0.00% | 105.91° | 0.00% | 30.28% | 97.36% | 0.040913 | 0.7857 |
| 9 | light_s06_r1_a115 | 58.17% | 0.00% | 115.21° | 0.00% | 26.90% | 96.87% | 0.052566 | 0.9790 |

## Automatic recommendation

`smooth_s12_r2_a115` is the best metric-balanced candidate in this sweep.
Confirm the winner and the default visually on palm-flip and finger-crossing frames before changing production defaults.
