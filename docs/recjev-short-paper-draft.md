# RecJev: Separating Decision Readout from Preference Learning in Movie Recommendation

## Abstract

Decision-oriented models return bounded scores that are easy to use in recommendation pipelines. Yet a well-formed decision score need not encode a particular user's preferences. We present **RecJev**, a reproducible study of this distinction using explicit ratings from MovieLens 1M. For each of 500 held-out users, we pair one movie rated 4–5 with one rated 1–2 and ask each model to score both movies using the same 20-rating history. The primary metric is whether the positive movie receives the higher score. We compare the Jev API, hosted language models, and frozen and task-tuned Qwen3 models under Open JEV's candidate-label readout. Frozen Qwen3-4B and Qwen3-8B reach 55.4% and 53.5% paired accuracy; task-aligned LoRA training raises them to 73.8% and 76.6% in matched one- and two-epoch runs, respectively. Substituting another user's history reduces the tuned 8B model from 76.6% to 68.6%, showing that its gain uses user information while substantial item-level signal remains. Training duration is model-dependent: a second epoch improves 8B but worsens 4B's Brier score. RecJev therefore separates the value of a decision interface from the acquisition of preference knowledge. These findings concern explicit ratings and unseen users within MovieLens; they do not establish click prediction or cross-domain transfer.

## 1. Introduction

Many recommendation decisions have a small, explicit answer space: given a user and a candidate item, should the system expect a favorable response? Decision-oriented interfaces are attractive here because they return a score directly, without generating and parsing a textual explanation. Jev is a hosted example of this interface, while Open JEV implements a related candidate-label readout over an open language model. Neither the interface nor the existence of a number, however, guarantees that the model has learned how a particular user's past ratings relate to a new movie. [TypeSafe's Jev announcement](https://typesafe.ai/blog/introducing-system-one-models-and-jev) and the [Open JEV implementation](https://github.com/zhihz/openjev) motivate studying this distinction empirically.

RecJev asks a narrow question: **when does a decision-style model use user history to distinguish a clearly liked movie from a clearly disliked one?** We focus on explicit high-versus-low ratings rather than claiming to estimate real-world click probability. A paired evaluation gives each test user one positive and one negative candidate, holds the user's history fixed, and judges the ordering of their scores. This makes ranking discrimination observable without interpreting a model's absolute score as calibrated real-world propensity.

Our experiments separate three factors that are often conflated. First, we compare frozen decision readouts with models adapted to the rating task. Second, we replace the real history with another user's history while keeping candidates and labels fixed, probing whether a trained model uses the user signal. Third, we compare model sizes and training epochs within the same protocol. The resulting pattern is informative: task alignment yields a large gain, a history intervention removes part of it, and neither scale nor an extra epoch guarantees improvement by itself.

This is an empirical benchmark and controlled analysis, rather than a claim of a new foundation-model architecture. Its contribution is a reproducible task construction, a protocol-matched tuning comparison, and a diagnostic for distinguishing user-history signal from candidate-level priors.

## 2. Task and evaluation

We use [MovieLens 1M](https://lenskit.grouplens.org/datasets/movielens/1m/). For each eligible user, ratings are ordered by timestamp. The first 80% supply the prior history; the final 20% supply held-out candidates. A positive candidate has a rating of 4 or 5; a negative candidate has a rating below 3; rating 3 is excluded. The test set contains 500 users, each with one positive and one negative candidate, for 1,000 single-candidate cases. Each request includes the user's 20 most recent ratings from the prior period and one candidate title. Both candidates for one user share the same history. This is a selected population of users with both types of held-out rating.

Training and validation users are disjoint from all 500 test users. The task-tuning set contains 3,296 users and 14,024 positive–negative pairs; validation contains 366 users and 1,506 pairs. Movies can recur across users, so this tests generalization to unseen users *within the same movie-rating domain*, not transfer to unseen items or another platform.

For each user, let \(p^+\) and \(p^-\) be the model's scores for the high- and low-rated movies. Paired accuracy awards 1 when \(p^+>p^-\), 0.5 for a tie, and 0 otherwise, then averages over users. Brier score averages \((p-y)^2\) over the 1,000 cases. Brier is useful for comparing score quality on this constructed, balanced task; these values are not validated estimates of population click or rating probabilities. We report 95% bootstrap intervals by resampling users, not individual cases.

## 3. Models and training

The controlled local study uses Qwen3-4B-Instruct-2507 and Qwen3-8B with fixed model revisions. Open JEV asks whether a user would rate a candidate 4–5 rather than 1–2, assigns the answer labels **A = No** and **B = Yes**, and applies a softmax to their next-token logits. The frozen models use this readout without rating-task training. We then train LoRA adapters on the same answer-label cross-entropy, leaving base weights frozen. Both sizes use rank 16, learning rate \(10^{-4}\), effective batch size 32, and seed 42. Each training run saves separate epoch checkpoints; evaluation uses exactly the same readout and test cases.

For context, RecJev also evaluates the hosted Jev Noul score and general-purpose language models asked to generate a JSON probability. These are useful reference points on the same cases, but their score-generation mechanisms, training access, deployment hardware, and request latency differ. We do not treat their numerical probabilities or latencies as directly interchangeable with local label-logit scores.

## 4. Results

**Task-aligned training matters more than the presence of a decision readout.** Table 1 contains the protocol-matched local comparison. The frozen readout is only modestly above chance in paired accuracy. Fine-tuning makes both sizes substantially more discriminative and improves Brier. The 8B two-epoch checkpoint reaches 76.6%; the matched first checkpoint reaches 72.6%, a gain of 4.0 percentage points (user-paired 95% interval +0.9 to +7.1). Brier falls by 0.0101 (interval −0.0166 to −0.0034). In contrast, a matched 4B second epoch leaves accuracy nearly unchanged and increases Brier. Thus checkpoint selection should consider both ranking and score quality rather than assuming more updates are always better.

**Table 1. Open JEV readout on 1,000 fixed MovieLens cases from 500 held-out users. Lower Brier is better.**

| Backbone | Training | Paired accuracy ↑ | Brier ↓ |
|---|---|---:|---:|
| Qwen3-4B | Frozen | 0.554 | 0.4810 |
| Qwen3-4B | LoRA, epoch 1 of matched two-epoch run | 0.738 | 0.2108 |
| Qwen3-4B | LoRA, epoch 2 of matched run | 0.735 | 0.2260 |
| Qwen3-8B | Frozen | 0.535 | 0.4608 |
| Qwen3-8B | LoRA, epoch 1 of matched two-epoch run | 0.726 | 0.2151 |
| Qwen3-8B | LoRA, epoch 2 of matched run | **0.766** | **0.2050** |

**The trained model uses history, but candidate priors remain influential.** Replacing each user's history with another user's history reduces the 8B epoch-2 checkpoint from 76.6% to 68.6% (paired interval for the −8.0-point change: −12.0 to −4.1). Its Brier rises from 0.2050 to 0.2401. In a separate 4B one-epoch run, the corresponding control reduced accuracy from 74.5% to 69.8%. The residual performance after history replacement cautions against describing the model as purely personalized; shared movie-level associations can still explain many pair orderings. History replacement also changes the input distribution, so it is a diagnostic rather than a precise causal decomposition.

**Hosted models provide context, not a controlled architecture comparison.** On the same 500-user test, Jev obtains 69.7% paired accuracy; GPT-5.6-sol and Claude Opus 5 obtain 74.7% and 75.3% using generated numeric probabilities. The tuned 8B checkpoint obtains 76.6%. Its +1.3-point difference from Claude has a user-paired interval crossing zero. Moreover, the local Qwen models receive domain-specific training that the hosted systems do not. These reference results demonstrate the range of attainable scores under available interfaces; they do not establish that one model architecture is universally superior.

## 5. Discussion and limitations

The central lesson is that **score format and preference competence must be evaluated separately**. A bounded, machine-readable decision score is operationally convenient, yet frozen label logits alone did not recover preferences well on this task. Domain-specific supervision improved discrimination, and the history intervention shows that some of the gain reflects user information. The distinct 4B and 8B epoch responses show why training duration should be selected empirically rather than transferred uncritically across sizes.

This evidence has clear boundaries. MovieLens supplies explicit ratings rather than impressions or clicks. We select users with both high and low held-out ratings, producing a balanced paired task rather than a realistic item-exposure distribution. Training and test users are disjoint, but movie titles and domain are shared. Results come from one train/validation split and one seed per model size; the bootstrap intervals do not measure run-to-run training variance. The local and hosted latency measurements include different hardware and network components, so they should not be combined into a speed claim. Finally, a conventional recommender baseline on this exact paired test and a second dataset would strengthen any broader recommendation claim.

## 6. Conclusion

RecJev shows that an Open JEV-style decision readout can become useful for explicit movie-rating discrimination after task-aligned tuning, and that a measurable part of the gain depends on the user's rating history. Its most defensible conclusion is conditional: within MovieLens, a decision interface is insufficient evidence of preference learning; training alignment and history dependence must be tested directly. Whether this finding extends to click behavior, new items, or other domains remains open.

## Before submission

1. Run a conventional ItemKNN/CF and popularity-only baseline on these exact 1,000 cases with the same user-history information.
2. Replicate the 4B and 8B tuning with additional seeds; report mean and spread, especially for the cross-size and epoch comparisons.
3. Add one separate recommendation dataset to test transfer beyond MovieLens and distinguish new-user from new-item generalization.
4. Keep hosted API accuracy, cost, and latency as a separately labeled reference table; report observed token usage and pricing assumptions.
