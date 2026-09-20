import argparse
import json
import os

from dotenv import load_dotenv

from .cases import load_cases
from .baselines import Popularity
from .evaluate import evaluate
from .models import Jev, OhMyGPT
from .movielens import MovieLens1M


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Candidate recommendation benchmark")
    parser.add_argument("--data", help="Extracted MovieLens 1M directory")
    parser.add_argument("--cases-file", help="Prepared JSONL cases; use with --limit for a fixed prefix")
    parser.add_argument("--limit", type=int, help="Run only the first N prepared cases")
    parser.add_argument("--resume", action="store_true", help="Append only missing cases after validating run settings")
    parser.add_argument("--provider", choices=["ohmygpt", "jev", "popularity"], required=True)
    parser.add_argument("--model", help="OhMyGPT model ID or Jev model ID")
    parser.add_argument("--omit-temperature", action="store_true", help="Use provider default for models that reject temperature=0")
    parser.add_argument("--disable-thinking", action="store_true", help="Send DeepSeek's thinking-disabled parameter")
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidates", type=int, default=10)
    parser.add_argument("--history-size", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--output", default="results/predictions.jsonl")
    parser.add_argument("--rates", default="rates.json", help="Published model rates snapshot, keyed by requested model ID")
    args = parser.parse_args()
    if args.top_k < 1 or args.top_k > args.candidates:
        parser.error("top-k must be between 1 and candidates")
    if args.cases_file:
        cases = load_cases(args.cases_file, args.limit)
    else:
        if not args.data:
            parser.error("--data or --cases-file is required")
        cases = MovieLens1M(args.data).cases(sample_size=args.sample_size, seed=args.seed,
                                              candidates=args.candidates, history_size=args.history_size)
        if args.limit is not None:
            cases = cases[:args.limit]
    if args.provider == "popularity":
        if not args.data:
            parser.error("--data is required for the popularity baseline")
        key = "local"
        model = Popularity(args.data)
    elif args.provider == "jev":
        key = os.getenv("JEV_API_KEY")
        model = Jev(key, os.getenv("JEV_ENDPOINT", "https://api.typesafe.ai/v1/systemone"),
                    args.model or os.getenv("JEV_MODEL", "jev-1.13.0"))
    else:
        key = os.getenv("OHMYGPT_API_KEY")
        if not args.model:
            parser.error("--model is required for OhMyGPT")
        model = OhMyGPT(key, os.getenv("OHMYGPT_BASE_URL", "https://api.ohmygpt.com/v1"),
                        args.model, temperature=None if args.omit_temperature else 0,
                        thinking=False if args.disable_thinking else None)
    if not key:
        parser.error(f"Missing {args.provider.upper()}_API_KEY in .env or environment")
    with open(args.rates) as file:
        rates = json.load(file)
    config = {"provider": args.provider, "temperature": None if args.omit_temperature else 0,
              "cases_file": args.cases_file, "sample_size": None if args.cases_file else args.sample_size,
              "seed": None if args.cases_file else args.seed}
    if args.disable_thinking:
        config["thinking"] = "disabled"
    if args.provider == "popularity":
        config["data"] = args.data
    print(json.dumps(evaluate(cases, model, args.top_k, args.output,
                              rate=rates.get(model.model), resume=args.resume,
                              run_config=config), indent=2))


if __name__ == "__main__":
    main()
