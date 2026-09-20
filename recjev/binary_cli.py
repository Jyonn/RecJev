"""Run the paired MovieLens high-rating probability benchmark."""

import argparse
import json
import os

from dotenv import load_dotenv

from .binary import JevProbability, LLMProbability, evaluate, load_cases, movielens_pairs, save_cases


def prepare() -> None:
    parser = argparse.ArgumentParser(description="Prepare one positive and one negative per user")
    parser.add_argument("--data", required=True)
    parser.add_argument("--sample-size", type=int, default=1000, help="Total examples; must be even")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--history-size", type=int, default=20)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--output", default="data/ml-1m/binary-1000-seed42.jsonl")
    args = parser.parse_args()
    cases = movielens_pairs(args.data, sample_size=args.sample_size, seed=args.seed,
                            history_size=args.history_size, train_fraction=args.train_fraction)
    checksum = save_cases(cases, args.output)
    print(json.dumps({"cases":len(cases),"users":len(cases)//2,"sha256":checksum,"output":args.output},indent=2))


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-file", required=True)
    parser.add_argument("--limit", type=int, default=200, help="Even number of examples")
    parser.add_argument("--provider", choices=["jev", "ohmygpt"], required=True)
    parser.add_argument("--model", help="Model ID; Jev defaults to JEV_MODEL")
    parser.add_argument("--omit-temperature", action="store_true")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--rates", default="rates.json")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cases = load_cases(args.cases_file, args.limit)
    if args.provider == "jev":
        key = os.getenv("JEV_API_KEY")
        model = JevProbability(key, os.getenv("JEV_ENDPOINT","https://api.typesafe.ai/v1/systemone"),
                               args.model or os.getenv("JEV_MODEL","jev-1.13.0"))
    else:
        key = os.getenv("OHMYGPT_API_KEY")
        if not args.model:
            parser.error("--model is required for OhMyGPT")
        model = LLMProbability(key,os.getenv("OHMYGPT_BASE_URL","https://api.ohmygpt.com/v1"),
                               args.model,args.omit_temperature,args.disable_thinking)
    if not key:
        parser.error(f"Missing {args.provider} key in .env")
    with open(args.rates) as file:
        rates = json.load(file)
    config = {"provider":args.provider,"cases_file":args.cases_file,
              "omit_temperature":args.omit_temperature,"disable_thinking":args.disable_thinking}
    print(json.dumps(evaluate(cases,model,args.output,rate=rates.get(model.model),
                              resume=args.resume,config=config),indent=2))


if __name__ == "__main__":
    main()
