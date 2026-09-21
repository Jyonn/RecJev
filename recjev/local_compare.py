"""Compare three probability protocols using the same local Qwen weights.

Run in an environment with CUDA PyTorch and the dependencies of zhihz/openjev.
The two modes share the exact local checkpoint and MovieLens case file.
"""

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

from .binary import BinaryCase, context, evaluate, load_cases


QUESTION = "Would this user rate the candidate movie 4 or 5 out of 5 rather than 1 or 2?"
YES = "Yes. This user would rate the candidate movie 4 or 5 out of 5."
NO = "No. This user would rate the candidate movie 1 or 2 out of 5."
GENERATION_SYSTEM = (
    "Estimate the probability that this user would rate the candidate movie "
    "4 or 5 out of 5 rather than 1 or 2. Return only a JSON object like "
    '{"probability": 0.5}. The probability must be between 0 and 1.'
)
NEXT_TOKEN_SYSTEM = (
    "Predict whether this user would rate the candidate movie 4 or 5 out of 5 "
    "rather than 1 or 2. Answer only Yes or No."
)


class LocalQwen:
    def __init__(self, repo: str | Path, assets: str | Path, mode: str, device: str,
                 result_model: str | None = None, max_new_tokens: int = 64):
        import torch

        self.torch = torch
        self.mode = mode
        self.model = result_model or {
            "direct": "openjev-qwen3-4b-direct",
            "generate": "qwen3-4b-generated",
            "next_token": "qwen3-4b-next-token",
        }[mode]
        self.last_usage = None
        self.max_new_tokens = max_new_tokens
        self.prefill_json = False
        self.last_response_model = None
        sys.path.insert(0, str(Path(repo).resolve()))
        from decisionmaking.instruction import InstructionBaseline

        self.engine = InstructionBaseline(assets, device=device, dtype=torch.float16)
        self.revision = self.engine.assets["model_revision"]
        self.last_response_model = self.revision
        if mode == "next_token":
            self.answer_token_ids = {}
            for answer in ("Yes", "No"):
                ids = self.engine.tokenizer.encode(answer, add_special_tokens=False)
                if len(ids) != 1:
                    raise ValueError(f"{answer!r} is not one token for this tokenizer: {ids}")
                self.answer_token_ids[answer] = ids[0]

    def predict(self, case: BinaryCase) -> float:
        self.last_usage = None
        self.last_response_model = self.revision
        if self.mode == "direct":
            from decisionmaking.schema import DecisionRequest

            request = DecisionRequest.model_validate({
                "state": context(case),
                "questions": [{"id": "liked", "type": "binary", "question": QUESTION,
                               "positive_id": "yes", "options": [
                                   {"id": "yes", "description": YES},
                                   {"id": "no", "description": NO}]}],
            })
            result = self.engine.decide(request)["results"][0]
            self.last_usage = {"input_tokens": result["input_tokens"], "output_tokens": 0}
            return float(result["probability"])

        system = NEXT_TOKEN_SYSTEM if self.mode == "next_token" else GENERATION_SYSTEM
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": context(case)}]
        if self.mode == "generate" and self.prefill_json:
            messages.append({"role": "assistant", "content": '{"probability":'})
        tokenizer = self.engine.tokenizer
        template_args = {"tokenize": False, "enable_thinking": False}
        if self.mode == "generate" and self.prefill_json:
            template_args["continue_final_message"] = True
        else:
            template_args["add_generation_prompt"] = True
        prompt = tokenizer.apply_chat_template(messages, **template_args)
        encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        inputs = {key: value.to(self.engine.device) for key, value in encoded.items()}
        if self.mode == "next_token":
            with self.torch.no_grad():
                logits = self.engine.model(**inputs).logits[0, -1]
            if self.engine.device.type == "cuda":
                self.torch.cuda.synchronize()
            pair = logits[[self.answer_token_ids["Yes"], self.answer_token_ids["No"]]].float()
            probability = self.torch.softmax(pair, dim=0)[0].item()
            self.last_usage = {"input_tokens": inputs["input_ids"].shape[1], "output_tokens": 0}
            return probability

        with self.torch.no_grad():
            output = self.engine.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                                do_sample=False,
                                                pad_token_id=tokenizer.eos_token_id)
        self.torch.cuda.synchronize() if self.engine.device.type == "cuda" else None
        generated = output[0, inputs["input_ids"].shape[1]:]
        self.last_usage = {"input_tokens": inputs["input_ids"].shape[1],
                           "output_tokens": len(generated)}
        content = tokenizer.decode(generated, skip_special_tokens=True)
        if self.prefill_json:
            content = '{"probability":' + content
        match = re.search(r"\{[\s\S]*?\}", content)
        if not match:
            raise ValueError(f"Expected JSON probability, got {content[:150]}")
        probability = float(json.loads(match.group())["probability"])
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Invalid probability")
        return probability


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-file", required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--openjev-repo", required=True)
    parser.add_argument("--assets", required=True, help="Open JEV frozen Qwen assets JSON")
    parser.add_argument("--mode", choices=["direct", "generate", "next_token"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--result-model", help="Model label stored in result rows")
    parser.add_argument("--max-new-tokens", type=int, default=64,
                        help="Generation budget for generate mode")
    parser.add_argument("--prefill-json", action="store_true",
                        help="Prefill the assistant JSON prefix in generate mode")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cases = load_cases(args.cases_file, args.limit)
    model = LocalQwen(args.openjev_repo, args.assets, args.mode, args.device,
                      result_model=args.result_model, max_new_tokens=args.max_new_tokens)
    model.prefill_json = args.prefill_json
    config = {"mode": args.mode, "model_revision": model.revision,
              "question": QUESTION, "positive_option": YES, "negative_option": NO,
              "system_prompt_sha256": hashlib.sha256(
                  (NEXT_TOKEN_SYSTEM if args.mode == "next_token" else GENERATION_SYSTEM).encode()
              ).hexdigest(),
              "max_new_tokens": args.max_new_tokens if args.mode == "generate" else None,
              "prefill_json": args.prefill_json if args.mode == "generate" else None,
              "openjev_protocol_sha256": model.engine.protocol_sha256}
    print(json.dumps(evaluate(cases, model, args.output, resume=args.resume,
                              config=config), indent=2))


if __name__ == "__main__":
    main()
