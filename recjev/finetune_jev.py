"""LoRA fine-tuning for Open JEV's exact two-option readout protocol."""

import argparse
import json
import random
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from .finetune_data import SYSTEM


class Examples(Dataset):
    def __init__(self, path: Path, tokenizer):
        self.rows = [json.loads(line) for line in path.open()]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": json.dumps(row["prompt"], ensure_ascii=False)}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if len(ids) > 4096:
            raise ValueError(f"Example exceeds Open JEV context limit: {row['case_id']}")
        return {"input_ids": ids, "label": int(row["label"]),
                "user_id": row["user_id"]}


def collate(batch, pad_id):
    length = max(len(row["input_ids"]) for row in batch)
    ids = torch.full((len(batch), length), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, row in enumerate(batch):
        n = len(row["input_ids"])
        ids[i, -n:] = torch.tensor(row["input_ids"])
        mask[i, -n:] = 1
    return {"input_ids": ids, "attention_mask": mask,
            "labels": torch.tensor([row["label"] for row in batch]),
            "users": [row["user_id"] for row in batch]}


def forward_scores(model, batch, label_ids, device):
    output = model(input_ids=batch["input_ids"].to(device),
                   attention_mask=batch["attention_mask"].to(device),
                   logits_to_keep=1, use_cache=False)
    # Open JEV sorts descriptions, so A = No and B = Yes.
    return output.logits[:, -1, label_ids].float()


@torch.no_grad()
def validate(model, loader, label_ids, device):
    model.eval()
    rows = []
    for batch in loader:
        logits = forward_scores(model, batch, label_ids, device)
        probabilities = logits.softmax(-1)[:, 1].cpu().tolist()
        rows.extend(zip(batch["users"], batch["labels"].tolist(), probabilities))
    by_user = {}
    for user, label, probability in rows:
        by_user.setdefault(user, {0: [], 1: []})[label].append(probability)
    wins = total = 0.0
    for pair in by_user.values():
        for pos, neg in zip(pair[1], pair[0]):
            wins += 1 if pos > neg else 0.5 if pos == neg else 0
            total += 1
    return {"examples": len(rows), "users": len(by_user),
            "pairwise_accuracy": wins / total,
            "brier": sum((probability-label)**2 for _, label, probability in rows)/len(rows),
            "mean_positive_probability": sum(p for _, y, p in rows if y)/sum(y for _, y, _ in rows),
            "mean_negative_probability": sum(p for _, y, p in rows if not y)/sum(not y for _, y, _ in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.grad_accum < 1:
        parser.error("epochs, batch size, and grad accumulation must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    assets = json.loads(args.assets.read_text())
    tokenizer = AutoTokenizer.from_pretrained(assets["model_path"], local_files_only=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    label_ids = [tokenizer.encode(x, add_special_tokens=False) for x in ("A", "B")]
    if any(len(ids) != 1 for ids in label_ids):
        raise ValueError("Open JEV option labels must be single tokens")
    label_ids = [ids[0] for ids in label_ids]
    train = Examples(args.data / "train.jsonl", tokenizer)
    validation = Examples(args.data / "validation.jsonl", tokenizer)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                        collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id))
    validation_loader = DataLoader(validation, batch_size=args.batch_size * 2,
                                   collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id))
    model = AutoModelForCausalLM.from_pretrained(
        assets["model_path"], local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa").to(device)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"base_model": assets["model_repo"], "base_revision": assets["model_revision"],
                "train_data_sha256": json.loads((args.data / "manifest.json").read_text())["files_sha256"],
                "epochs": args.epochs, "batch_size": args.batch_size,
                "grad_accum": args.grad_accum, "learning_rate": args.learning_rate,
                "seed": args.seed, "lora_rank": 16, "lora_alpha": 32,
                "lora_dropout": 0.05, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "protocol": "Open JEV two-label next-token cross entropy"}
    (args.output / "training-manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for step, batch in enumerate(loader, 1):
            logits = forward_scores(model, batch, label_ids, device)
            loss = torch.nn.functional.cross_entropy(logits, batch["labels"].to(device))
            (loss / args.grad_accum).backward()
            running += loss.item()
            if step % args.grad_accum == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if step % 100 == 0:
                print(json.dumps({"epoch": epoch+1, "step": step,
                                  "total_steps": len(loader), "mean_loss": running/step}), flush=True)
        metrics = validate(model, validation_loader, label_ids, device)
        print(json.dumps({"epoch": epoch+1, "validation": metrics}), flush=True)
        if metrics["pairwise_accuracy"] > best:
            best = metrics["pairwise_accuracy"]
            model.save_pretrained(args.output / "adapter")
            (args.output / "validation.json").write_text(json.dumps(metrics, indent=2)+"\n")


if __name__ == "__main__":
    main()
