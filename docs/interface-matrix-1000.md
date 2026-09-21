# Decision-interface matrix on 1,000 MovieLens rating cases

All twelve cells use the same 500 users and one positive/negative item pair per user. The frozen and tuned cells for a given backbone use the same base model, but tuned 4B is the independent one-epoch adapter and tuned 8B is epoch 2 of the matched two-epoch run. All result files contain 1,000 matching case IDs and labels, with zero errors. The original 200-case files match the first 200 rows of the new frozen 4B files exactly.

| Backbone | State | Open JEV direct | Generated number | Yes/No next token |
|---|---|---:|---:|---:|
| Qwen3-4B | Frozen | 55.4% | 59.0% | 63.5% |
| Qwen3-4B | Tuned, 1 epoch | 74.5% | 71.7% | 70.5% |
| Qwen3-8B | Frozen | 53.5% | 60.1% | 60.9% |
| Qwen3-8B | Tuned, 2 epochs | 76.6% | 73.0% | 72.6% |

## User-paired accuracy differences

Intervals below use 10,000 bootstrap samples of the 500 users, seed 42. Ties contribute 0.5.

| Contrast | Difference | 95% interval |
|---|---:|---:|
| 4B frozen: next token − direct | +8.1 points | [+3.8, +12.3] |
| 8B frozen: next token − direct | +7.4 points | [+2.8, +11.9] |
| 4B tuned: direct − next token | +4.0 points | [+0.8, +7.2] |
| 8B tuned: direct − next token | +4.0 points | [+1.1, +6.9] |
| 4B interaction: change in direct − next-token gap after tuning | +12.1 points | [+6.9, +17.2] |
| 8B interaction: change in direct − next-token gap after tuning | +11.4 points | [+6.4, +16.4] |

The interaction compares two checkpoints trained with different objectives: the tuned adapter learned Open JEV's option-label target. It demonstrates alignment between training and readout, not a pure output-interface effect. The three readouts also use different question wording and answer formats. The experiment is restricted to MovieLens explicit high-versus-low ratings and one training seed per backbone.

## Classical baselines on the exact pairs

Popularity, ItemKNN, and UserKNN use positive ratings from each user's first 80% timestamp-ordered ratings only. The test user's own ratings are excluded as neighbors in UserKNN. Scores are not probabilities, so Brier is inapplicable.

| Baseline | Paired accuracy |
|---|---:|
| Popularity | 65.2% |
| ItemKNN | 63.3% |
| UserKNN | 61.0% |

Official Jev scored 69.7% on the same 500 users. Its difference from popularity was +4.5 points with a user-paired 95% interval of [−0.4, +9.4] points.

## Result locations

- Frozen 4B: `results/binary-1000-local/`.
- Tuned 4B direct: `results/finetune/qwen3-4b-openjev-seed42/robustness/lora-original.jsonl`; other tuned 4B readouts: `results/binary-1000-local/`.
- Frozen 8B direct: `results/finetune/qwen3-8b-openjev-seed42/base-test.jsonl`; other frozen 8B readouts: `results/binary-1000-local/`.
- Tuned 8B direct: `results/finetune/qwen3-8b-openjev-seed42-2epochs/epoch-2-original.jsonl`; other tuned 8B readouts: `results/binary-1000-local/`.
- Classical baselines: `results/binary-1000-baselines/`, generated with `python -m recjev.binary_baselines --data data/ml-1m --cases-file data/ml-1m/binary-1000-seed42.jsonl --output-dir results/binary-1000-baselines`.

The result directories are ignored by Git and retained locally and on the research server. The source dataset is MovieLens 1M, with fixed test sample seed 42.
