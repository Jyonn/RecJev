import argparse
import json
import os

from dotenv import load_dotenv

from .evaluate import evaluate
from .models import Jev, OhMyGPT
from .movielens import MovieLens1M


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Candidate recommendation benchmark")
    parser.add_argument("--data", required=True, help="Extracted MovieLens 1M directory")
    parser.add_argument("--provider", choices=["ohmygpt", "jev"], required=True)
    parser.add_argument("--model", help="OhMyGPT model ID or Jev model ID")
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidates", type=int, default=10)
    parser.add_argument("--history-size", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--output", default="results/predictions.jsonl")
    args = parser.parse_args()
    if args.top_k < 1 or args.top_k > args.candidates:
        parser.error("top-k must be between 1 and candidates")
    cases = MovieLens1M(args.data).cases(sample_size=args.sample_size, seed=args.seed,
                                          candidates=args.candidates, history_size=args.history_size)
    if args.provider == "jev":
        key = os.getenv("JEV_API_KEY")
        model = Jev(key, os.getenv("JEV_ENDPOINT", "https://api.typesafe.ai/v1/systemone"),
                    args.model or os.getenv("JEV_MODEL", "jev-1.13.0"))
    else:
        key = os.getenv("OHMYGPT_API_KEY")
        if not args.model:
            parser.error("--model is required for OhMyGPT")
        model = OhMyGPT(key, os.getenv("OHMYGPT_BASE_URL", "https://api.ohmygpt.com/v1"), args.model)
    if not key:
        parser.error(f"Missing {args.provider.upper()}_API_KEY in .env or environment")
    print(json.dumps(evaluate(cases, model, args.top_k, args.output), indent=2))


if __name__ == "__main__":
    main()
