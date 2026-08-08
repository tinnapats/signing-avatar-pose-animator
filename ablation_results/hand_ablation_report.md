# Hand Pipeline Ablation Study

- Generated: `2026-08-04T14:13:43.935335+00:00`
- Signs: `abdomen, hello, love`
- Runtime: `36.66 seconds`
- Visible-hand threshold: `0.18`
- Joint violation threshold: `115.0 degrees`

## Summary

| Variant | Coverage | Bone span (max) ↓ | Max bend ↓ | Joint violations ↓ | Topology errors ↓ | Motion retained ↑ | Shape jerk ↓ | Deviation from full ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 47.85% | 0.00% | 115.00° | 0.00% | 7.23% | 99.12% | 0.031943 | 0.000000 |
| no_3d_slerp | 47.85% | 0.00% | 116.16° | 0.02% | 7.23% | 98.83% | 0.048985 | 0.006242 |
| no_gap_repair | 45.87% | 0.00% | 115.00° | 0.00% | 7.25% | 99.38% | 0.026676 | 0.007147 |
| no_hand_smoothing | 47.85% | 0.00% | 115.00° | 0.00% | 8.96% | 95.03% | 0.071378 | 0.010968 |
| no_bone_stabilizer | 47.85% | 0.00% | 115.00° | 0.00% | 7.23% | 98.83% | 0.032657 | 0.000632 |
| no_fixed_length | 47.85% | 245.29% | 177.91° | 13.74% | 10.40% | 98.25% | 0.029889 | 0.101035 |
| no_joint_limit | 47.85% | 0.00% | 175.00° | 13.74% | 7.51% | 98.54% | 0.030678 | 0.008785 |
| raw_hand_baseline | 45.40% | 243.14% | 178.20° | 14.21% | 9.45% | 94.34% | 0.065488 | 0.096151 |

## Component effects

- Full system: bone span `0.00%`, joint violations `0.00%`, motion retained `99.12%`.
- Without 3D SLERP: bone span `0.00%`, max bend `116.16°`, motion `98.83%`, jerk `0.048985`.
- Without gap repair: bone span `0.00%`, max bend `115.00°`, motion `99.38%`, jerk `0.026676`.
- Without hand smoothing: bone span `0.00%`, max bend `115.00°`, motion `95.03%`, jerk `0.071378`.
- Without bone stabilizer: bone span `0.00%`, max bend `115.00°`, motion `98.83%`, jerk `0.032657`.
- Without fixed-length articulation: bone span `245.29%`, max bend `177.91°`, motion `98.25%`, jerk `0.029889`.
- Without joint limit: bone span `0.00%`, max bend `175.00°`, motion `98.54%`, jerk `0.030678`.
- Without raw baseline: bone span `243.14%`, max bend `178.20°`, motion `94.34%`, jerk `0.065488`.

## Interpretation notes

- Bone span uses the robust `(P95 - P05) / median` length of each 3D kinematic bone; the table reports the worst bone across signs.
- Shape jerk is the third temporal difference of wrist-relative landmarks, normalized by palm size.
- Topology errors are 2D validation warnings; some may be real perspective overlap rather than anatomical failure.
- This pilot covers three signs. Run the same script with more dataset words before reporting a final statistical conclusion.
