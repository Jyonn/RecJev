# RecJev

A small benchmark for **candidate based movie recommendation** with Jev and models reached through OhMyGPT. The first release uses MovieLens 1M and a leave last positive out task. It reports Hit@K, NDCG@K, mean request latency, and one JSONL prediction record per case.

## Install and prepare data

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

Download [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) and extract it so `data/ml-1m/movies.dat` and `data/ml-1m/ratings.dat` exist. The dataset is ignored by Git. Fill the relevant API key in `.env`; endpoints and Jev model can be changed there. Do not commit `.env`.

To export and inspect the fixed 1000-case evaluation set without calling a model:

```bash
recjev-prepare --data data/ml-1m --sample-size 1000 --seed 42 --candidates 10 --history-size 20 --output data/ml-1m/eval-1000-seed42.jsonl
```

The command prints the JSONL SHA-256 checksum. The file stays local because `data/` is ignored by Git. Use the same sampling parameters for each model run.

For the more discriminating **10-choice Top-1** experiment, prepare cases with one rating of at least 4 and nine ratings below 3 from the same user's held-out portion:

```bash
recjev-prepare --data data/ml-1m --sample-size 1000 --seed 42 --candidates 10 --history-size 20 --negatives rated-low --train-fraction 0.8 --output data/ml-1m/eval-rated-low-1000-seed42.jsonl
```

For each user, the first 80% of timestamp-ordered ratings form the training prefix. The last 20% supply a high-rated positive and nine sampled low-rated negatives. A user needs enough held-out ratings to qualify; with MovieLens 1M, this yields at least 1000 users. This is a sampled candidate task, not full-catalog recommendation. The case JSONL includes ratings for the last 20 training-history movies, while candidate ratings stay hidden. Record the split and negative strategy alongside result files.

## Run

```bash
recjev --data data/ml-1m --provider ohmygpt --model gpt-4o --sample-size 1000 --seed 42 --candidates 10 --top-k 3 --output results/gpt.jsonl
recjev --data data/ml-1m --provider jev --sample-size 1000 --seed 42 --candidates 10 --top-k 3 --output results/jev.jsonl
```

For a fixed prefix of the exported evaluation set, use the same file and limit for each model:

```bash
recjev --cases-file data/ml-1m/eval-1000-seed42.jsonl --limit 200 --provider jev --top-k 3 --output results/benchmark200/jev.jsonl --resume
recjev --cases-file data/ml-1m/eval-1000-seed42.jsonl --limit 200 --provider ohmygpt --model gpt-4o --top-k 3 --output results/benchmark200/gpt-4o.jsonl --resume
recjev --cases-file data/ml-1m/eval-1000-seed42.jsonl --limit 200 --provider popularity --data data/ml-1m --top-k 3 --output results/benchmark200/popularity.jsonl --resume
```

Run the new 10-choice Top-1 experiment on the same fixed first 200 cases:

```bash
recjev --cases-file data/ml-1m/eval-rated-low-1000-seed42.jsonl --limit 200 --data data/ml-1m --provider popularity --negatives rated-low --top-k 1 --output results/rated-low-200/popularity.jsonl --resume
recjev --cases-file data/ml-1m/eval-rated-low-1000-seed42.jsonl --limit 200 --data data/ml-1m --provider itemknn --negatives rated-low --top-k 1 --output results/rated-low-200/itemknn.jsonl --resume
recjev --cases-file data/ml-1m/eval-rated-low-1000-seed42.jsonl --limit 200 --data data/ml-1m --provider userknn --negatives rated-low --top-k 1 --output results/rated-low-200/userknn.jsonl --resume
recjev --cases-file data/ml-1m/eval-rated-low-1000-seed42.jsonl --limit 200 --provider jev --negatives rated-low --top-k 1 --output results/rated-low-200/jev.jsonl --resume
recjev --cases-file data/ml-1m/eval-rated-low-1000-seed42.jsonl --limit 200 --provider ohmygpt --model gpt-4o --negatives rated-low --top-k 1 --output results/rated-low-200/gpt-4o.jsonl --resume
```

`--resume` validates the selected cases, model, Top-K, rate snapshot, and run settings against a companion `.meta.json` file, then skips completed rows. Each row is flushed to disk immediately. A network or provider HTTP failure stops the run and leaves that case pending for the next invocation. A complete but malformed model answer is recorded as a failed case. Without `--resume`, the command refuses to overwrite existing results.

For `--negatives rated-low`, all local baselines train only on the first `--train-fraction` of each user's ratings. Popularity counts high ratings, ItemKNN scores candidates by cosine similarity to the user's liked items in the displayed history, and UserKNN uses the 50 most similar users based on that displayed history and positive training interactions. They need no API key. Their local runtime is reported separately from hosted model latency. Pass the same `--negatives` and `--train-fraction` flags for every run, including when loading `--cases-file`, so the result manifest identifies the protocol.

Use the exact model ID available to your OhMyGPT account for GPT, Qwen, Llama, or DeepSeek. A small smoke run uses `--sample-size 5`. `--top-k 1` is Top-1; larger values request an ordered Top-K list. The same seed, sample size, candidate count, and history size produce the same cases across runs. The CLI currently uses MovieLens 1M only; no data is downloaded automatically.

Some models reject `temperature=0`. Add `--omit-temperature` for those models; this uses the provider default and should be recorded when comparing results.
DeepSeek Flash enables thinking by default; `--disable-thinking` sends its documented `thinking: disabled` setting and records that setting in the run metadata. Use a separate result file for each configuration.

## Protocol and extension

`recjev/protocol.py` defines `Case`, `DatasetAdapter`, and `Recommender`. A dataset adapter implements `cases(sample_size, seed, candidates, history_size)` and returns cases with stable IDs, a history, candidates, and a held out positive. Add another adapter and select it in `cli.py`. A model implements `rank(case, top_k)` and returns distinct candidate IDs in ranked order. Add its configuration in `.env.example` and construction in `cli.py`.

The original `unseen` protocol takes each user's last rating of at least 4 as positive, earlier ratings as history, and samples unobserved movies as candidates. The `rated-low` protocol uses the user's first 80% of ratings for history and local baseline training, then takes the last high rating in the held-out 20% as positive and samples ratings below 3 as negatives. Cases lacking enough candidates are skipped. The 3-star ratings are excluded from negative sampling. Both are sampled candidate tasks; neither measures full-catalog recommendation or future user satisfaction. `Case.constraints` reserves a field for a future constraint aware task. Jev returns a Choice distribution; Top-K is the highest probability IDs. The LLM is asked for an ordered list directly, so the ranking mechanisms differ.

For publication, pin model IDs, retain the output JSONL and command line parameters, and report transport failures separately from malformed model answers. Transport failures leave the case pending for `--resume`; malformed answers count as misses.
Each prediction record also stores the requested model, returned model, the provider's raw `usage` token counts (when supplied), and the published rate snapshot from `rates.json`. Update that file before a new experiment or pass another file with `--rates`. Rates use the listed currency per million tokens; they are not proof of the account's actual charge. DeepSeek has peak/off-peak and cached-input rates, so its rate entry records all tiers without selecting one. A missing `usage` or rate is stored as `null`.

`latency_s` is measured locally with a monotonic clock around `model.rank()`: it includes network transit, provider processing, response parsing, and validation. It is **not** a latency field supplied by the API. For a single-process sequential run, `mean_latency_s` averages successful requests only; it is `null` when all requests fail. Failed attempts keep their own `latency_s` in the JSONL file.

## Test

```bash
python -m unittest discover -s tests -v
```

## API references

- [OhMyGPT API reference](https://docs.ohmygpt.com/docs/api)
- [TypeSafe Jev API](https://docs.typesafe.ai/)
