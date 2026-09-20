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

## Run

```bash
recjev --data data/ml-1m --provider ohmygpt --model gpt-4o --sample-size 1000 --seed 42 --candidates 10 --top-k 3 --output results/gpt.jsonl
recjev --data data/ml-1m --provider jev --sample-size 1000 --seed 42 --candidates 10 --top-k 3 --output results/jev.jsonl
```

Use the exact model ID available to your OhMyGPT account for GPT, Qwen, Llama, or DeepSeek. A small smoke run uses `--sample-size 5`. `--top-k 1` is Top-1; larger values request an ordered Top-K list. The same seed, sample size, candidate count, and history size produce the same cases across runs. The CLI currently uses MovieLens 1M only; no data is downloaded automatically.

Some models reject `temperature=0`. Add `--omit-temperature` for those models; this uses the provider default and should be recorded when comparing results.

## Protocol and extension

`recjev/protocol.py` defines `Case`, `DatasetAdapter`, and `Recommender`. A dataset adapter implements `cases(sample_size, seed, candidates, history_size)` and returns cases with stable IDs, a history, candidates, and a held out positive. Add another adapter and select it in `cli.py`. A model implements `rank(case, top_k)` and returns distinct candidate IDs in ranked order. Add its configuration in `.env.example` and construction in `cli.py`.

MovieLens cases take each user's last rating of at least 4 as the positive, earlier rated movies as history, and uniformly sample negatives from movies the user never rated. Cases lacking enough negatives are skipped. The dataset gives no true negative labels, so Hit@K measures retrieval of a sampled held out positive, not general preference quality. `Case.constraints` reserves a field for a future constraint aware task. Jev returns a Choice distribution; Top-K is the highest probability IDs. The LLM is asked for an ordered list directly, so the ranking mechanisms differ. This should be stated when reporting comparative results.

This minimal release records latency, but not cost because billing rates depend on the selected provider/model and account. API failures are retained as failed cases in the prediction file and count as misses. For publication, pin model IDs, record pricing separately, and retain the output JSONL and command line parameters.
Each prediction record also stores the requested model, returned model, the provider's raw `usage` token counts (when supplied), and the published rate snapshot from `rates.json`. Update that file before a new experiment or pass another file with `--rates`. Rates use the listed currency per million tokens; they are not proof of the account's actual charge. DeepSeek has peak/off-peak and cached-input rates, so its rate entry records all tiers without selecting one. A missing `usage` or rate is stored as `null`.

`latency_s` is measured locally with a monotonic clock around `model.rank()`: it includes network transit, provider processing, response parsing, and validation. It is **not** a latency field supplied by the API. For a single-process sequential run, `mean_latency_s` averages successful requests only; it is `null` when all requests fail. Failed attempts keep their own `latency_s` in the JSONL file.

## Test

```bash
python -m unittest discover -s tests -v
```

## API references

- [OhMyGPT API reference](https://docs.ohmygpt.com/docs/api)
- [TypeSafe Jev API](https://docs.typesafe.ai/)
