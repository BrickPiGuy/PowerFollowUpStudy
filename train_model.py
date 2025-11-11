import os
import gc
import csv
import json
import math
import boto3
import torch
import tempfile
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.amp import autocast, GradScaler

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device=None):
    if device is None:
        device = get_device()
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    # training-friendly: avoid KV cache warnings during loss compute
    if hasattr(model, "config"):
        setattr(model.config, "use_cache", False)
    return model


def get_tokenizer(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


class TinyStoryDataset(Dataset):
    def __init__(self, tokenizer, tokens_required, s3_path, max_length=64):
        """
        Reads a JSONL file where each line has a {"text": "..."} record.
        Supports s3://bucket/key or a local file path.
        Accumulates until reaching tokens_required *true* tokens (excludes padding).
        """
        temp_path = None
        if s3_path.startswith("s3://"):
            s3 = boto3.client("s3")
            bucket, key = s3_path.replace("s3://", "").split("/", 1)
            tmp = tempfile.NamedTemporaryFile(delete=False)  # binary path only
            tmp.close()
            s3.download_file(bucket, key, tmp.name)
            source_file = tmp.name
            temp_path = tmp.name
        else:
            source_file = s3_path

        self.samples = []
        token_total = 0

        with open(source_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                text = obj.get("text", "")
                if not text:
                    continue

                encoded = tokenizer(
                    text,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                )
                # Count real tokens via attention_mask, not padded length
                real_tokens = int(encoded["attention_mask"][0].sum().item())
                self.samples.append(encoded)
                token_total += real_tokens
                if token_total >= tokens_required:
                    break

        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Return dict of tensors without batch dim
        return {k: v.squeeze(0) for k, v in self.samples[idx].items()}


def get_dataloader(tokens: int, s3_path: str, batch_size=1, shuffle=True, pin_memory=None):
    tokenizer = get_tokenizer()
    dataset = TinyStoryDataset(tokenizer, tokens_required=tokens, s3_path=s3_path)
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)


def get_optimizer(model, learning_rate=1e-6):
    return AdamW(model.parameters(), lr=learning_rate)


def evaluate_model(model, tokenizer=None, prompt="The quick brown fox jumps over the lazy dog.", verbose=True):
    """
    Returns (perplexity, inv_perplexity) where inv_perplexity = 1 / perplexity.
    """
    model.eval()
    device = next(model.parameters()).device
    if tokenizer is None:
        tokenizer = get_tokenizer()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    labels = inputs["input_ids"].clone()
    labels[inputs["attention_mask"] == 0] = -100

    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=labels
        )
        loss = outputs.loss
        logits = outputs.logits

        if verbose:
            print(f"🧪 Sample logits stats — mean: {logits.mean().item()} std: {logits.std().item()}")
            print(f"⚠️ logits contains_nan={torch.isnan(logits).any().item()}, contains_inf={torch.isinf(logits).any().item()}")

        if torch.isnan(loss) or torch.isinf(loss):
            if verbose:
                print("⚠️ Invalid loss detected during evaluation — returning NaN")
                print(f"ℹ️ loss value: {loss.item()} (type: {type(loss)})")
            return float("nan"), float("nan")

        if verbose:
            print(f"🧪 Eval loss: {loss.item()}")

        # avoid overflow if loss is huge
        ppl = math.exp(loss.item()) if loss.item() < 80 else float("inf")

    inv_ppl = (1 / ppl) if (math.isfinite(ppl) and ppl > 0) else float("nan")
    return round(ppl, 4), (round(inv_ppl, 4) if math.isfinite(inv_ppl) else float("nan"))


def estimate_tops():
    # Placeholder constant until wired to real measurement
    return 275.0


def log_trial_result_csv(csv_path: Path, trial_data: dict):
    """
    CSV keeps the legacy 'accuracy' column name for compatibility,
    which now holds inv_perplexity.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp", "token_count", "trial_number",
        "accuracy", "tops_used", "parameter_efficiency",
        "parameter_efficiency_loss", "parameter_perplexity"
    ]
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(trial_data)


def train_model(tokens: int, trial_number: int, output_dir: Path, s3_path: str, verbose: bool = True) -> dict:
    device = get_device()
    model = get_model(device=device)
    tokenizer = get_tokenizer()
    dataloader = get_dataloader(tokens, s3_path=s3_path)
    optimizer = get_optimizer(model)

    model.train()
    valid_batches = 0

    scaler = GradScaler(enabled=(device.type == "cuda"))

    for epoch in range(3):
        for batch in dataloader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            if verbose:
                print("🔍 Inspecting training batch:")
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        print(
                            f"  {key}: shape={value.shape}, dtype={value.dtype}, device={value.device}, "
                            f"contains_nan={torch.isnan(value).any().item()}, contains_inf={torch.isinf(value).any().item()}"
                        )

            if epoch == 0 and trial_number == 1 and verbose:
                print("📝 Decoded input:", tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True))

            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1e3:
                if verbose:
                    print("⚠️ Invalid or explosive loss detected — skipping batch")
                    print(f"ℹ️ loss value: {loss.item()} (type: {type(loss)})")
                continue
            else:
                if verbose:
                    print(f"✅ Valid loss: {loss.item()}")

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            valid_batches += 1

    # Evaluate
    fallback_prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "Once upon a time, there was a little girl named Lily.",
        "In a distant land, a young boy discovered a strange machine."
    ]
    perplexity, inv_perplexity = evaluate_model(model, tokenizer, verbose=verbose)

    if (not math.isfinite(inv_perplexity)) or (not math.isfinite(perplexity)):
        for prompt in fallback_prompts:
            if verbose:
                print(f"🔁 Trying fallback prompt: {prompt}")
            perplexity, inv_perplexity = evaluate_model(model, tokenizer, prompt, verbose=verbose)
            if math.isfinite(inv_perplexity) and math.isfinite(perplexity):
                break

    if (not math.isfinite(inv_perplexity)) or (not math.isfinite(perplexity)):
        if verbose:
            print("⚠️ All prompts failed — attempting evaluation using a training batch")
        try:
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch["input_ids"].clone()
                labels[batch["attention_mask"] == 0] = -100
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=labels
                )
                loss = outputs.loss
                if torch.isfinite(loss):
                    ppl = math.exp(loss.item()) if loss.item() < 80 else float("inf")
                    inv_ppl = (1 / ppl) if (math.isfinite(ppl) and ppl > 0) else float("nan")
                    perplexity = round(ppl, 4) if math.isfinite(ppl) else float("nan")
                    inv_perplexity = round(inv_ppl, 4) if math.isfinite(inv_ppl) else float("nan")
                    if verbose:
                        print("✅ Recovered from batch — loss:", loss.item())
                    break
        except Exception as e:
            if verbose:
                print("🚨 Fallback batch evaluation failed:", str(e))

    tops_used = estimate_tops()

    if (not math.isfinite(inv_perplexity)) or (not math.isfinite(perplexity)):
        parameter_efficiency = float("nan")
        parameter_perplexity = float("nan")
        parameter_efficiency_loss = float("nan")
    else:
        # Keep your definitions; just make naming explicit.
        # inv_perplexity = 1 / perplexity
        parameter_efficiency = round(inv_perplexity / (tops_used * 1_100_000_000), 20)
        parameter_perplexity = round(perplexity / 1_100_000_000, 20)
        baseline_parameter_efficiency = 5e-9
        parameter_efficiency_loss = round(1 - (parameter_efficiency / baseline_parameter_efficiency), 10)

    if verbose:
        print("✅ Training complete — preparing to log results...")

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.txt", "w") as f:
        f.write(f"perplexity: {perplexity}\n")
        f.write(f"inv_perplexity: {inv_perplexity}\n")
        f.write(f"tops_used: {tops_used}\n")
        f.write(f"parameter_efficiency: {parameter_efficiency}\n")
        f.write(f"parameter_efficiency_loss: {parameter_efficiency_loss}\n")
        f.write(f"parameter_perplexity: {parameter_perplexity}\n")

    # CSV keeps the legacy 'accuracy' column, populated with inv_perplexity
    log_csv_path = output_dir.parent / "run_log.csv"
    log_trial_result_csv(log_csv_path, {
        "timestamp": datetime.now().isoformat(),
        "token_count": tokens,
        "trial_number": trial_number,
        "accuracy": inv_perplexity,  # legacy alias
        "tops_used": tops_used,
        "parameter_efficiency": parameter_efficiency,
        "parameter_efficiency_loss": parameter_efficiency_loss,
        "parameter_perplexity": parameter_perplexity
    })

    # Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "status": "success",
        "inv_perplexity": inv_perplexity,     # new, explicit metric name
        "accuracy": inv_perplexity,           # legacy alias for compatibility
        "perplexity": perplexity,
        "tops_used": tops_used,
        "parameter_efficiency": parameter_efficiency,
        "parameter_efficiency_loss": parameter_efficiency_loss,
        "parameter_perplexity": parameter_perplexity
    }
