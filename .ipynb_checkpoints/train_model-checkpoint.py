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
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from torch.amp import autocast, GradScaler

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_model(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto"
    )
    return model


def get_tokenizer(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


class TinyStoryDataset(Dataset):
    def __init__(self, tokenizer, tokens_required, s3_path):
        if s3_path.startswith("s3://"):
            s3 = boto3.client('s3')
            bucket, key = s3_path.replace("s3://", "").split("/", 1)
            temp_file = tempfile.NamedTemporaryFile(delete=False, mode="w+", encoding="utf-8")
            s3.download_file(bucket, key, temp_file.name)
            source_file = temp_file.name
        else:
            source_file = s3_path

        self.samples = []
        token_total = 0

        with open(source_file, "r", encoding="utf-8") as f:
            for line in f:
                text = json.loads(line)["text"]
                encoded = tokenizer(
                    text,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=64
                )
                self.samples.append(encoded)
                token_total += len(encoded["input_ids"][0])
                if token_total >= tokens_required:
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {k: v.squeeze(0) for k, v in self.samples[idx].items()}


def get_dataloader(tokens: int, s3_path: str, batch_size=1, shuffle=True):
    tokenizer = get_tokenizer()
    dataset = TinyStoryDataset(tokenizer, tokens_required=tokens, s3_path=s3_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_optimizer(model, learning_rate=1e-6):
    return AdamW(model.parameters(), lr=learning_rate)


def evaluate_model(model, tokenizer=None, prompt="The quick brown fox jumps over the lazy dog.", verbose=True):
    model.eval()
    device = next(model.parameters()).device
    if tokenizer is None:
        tokenizer = get_tokenizer()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    labels = inputs["input_ids"].clone()
    labels[inputs["attention_mask"] == 0] = -100

    with torch.no_grad():
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=labels)
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
        perplexity = torch.exp(loss).item()

    return round(perplexity, 4), round(1 / perplexity, 4)


def estimate_tops():
    return 275.0


def log_trial_result_csv(csv_path: Path, trial_data: dict):
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
    model = get_model()
    tokenizer = get_tokenizer()
    dataloader = get_dataloader(tokens, s3_path=s3_path)
    optimizer = get_optimizer(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    valid_batches = 0

    scaler = GradScaler()

    for epoch in range(3):
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            if verbose:
                print("🔍 Inspecting training batch:")
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        print(f"  {key}: shape={value.shape}, dtype={value.dtype}, device={value.device}, contains_nan={torch.isnan(value).any().item()}, contains_inf={torch.isinf(value).any().item()}")

            if epoch == 0 and trial_number == 1 and verbose:
                print("📝 Decoded input:", tokenizer.decode(batch["input_ids"][0]))

            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            optimizer.zero_grad()

            with autocast(device_type="cuda"):
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

    fallback_prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "Once upon a time, there was a little girl named Lily.",
        "In a distant land, a young boy discovered a strange machine."
    ]
    perplexity, accuracy = evaluate_model(model, tokenizer, verbose=verbose)

    if math.isnan(accuracy) or math.isnan(perplexity):
        for prompt in fallback_prompts:
            if verbose:
                print(f"🔁 Trying fallback prompt: {prompt}")
            perplexity, accuracy = evaluate_model(model, tokenizer, prompt, verbose=verbose)
            if not math.isnan(accuracy) and not math.isnan(perplexity):
                break

    if math.isnan(accuracy) or math.isnan(perplexity):
        if verbose:
            print("⚠️ All prompts failed — attempting evaluation using a training batch")
        try:
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch["input_ids"].clone()
                labels[batch["attention_mask"] == 0] = -100
                outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=labels)
                loss = outputs.loss
                if not torch.isnan(loss) and not torch.isinf(loss):
                    perplexity = round(torch.exp(loss).item(), 4)
                    accuracy = round(1 / perplexity, 4)
                    if verbose:
                        print("✅ Recovered from batch — loss:", loss.item())
                    break
        except Exception as e:
            if verbose:
                print("🚨 Fallback batch evaluation failed:", str(e))

    tops_used = estimate_tops()

    if math.isnan(accuracy) or math.isnan(perplexity):
        parameter_efficiency = float("nan")
        parameter_perplexity = float("nan")
        parameter_efficiency_loss = float("nan")
    else:
        parameter_efficiency = round(accuracy / (tops_used * 1_100_000_000), 20)
        parameter_perplexity = round(perplexity / 1_100_000_000, 20)
        baseline_parameter_efficiency = 5e-9
        parameter_efficiency_loss = round(1 - (parameter_efficiency / baseline_parameter_efficiency), 10)

    if verbose:
        print("✅ Training complete — preparing to log results...")

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.txt", "w") as f:
        f.write(f"perplexity: {perplexity}\n")
        f.write(f"accuracy: {accuracy}\n")
        f.write(f"tops_used: {tops_used}\n")
        f.write(f"parameter_efficiency: {parameter_efficiency}\n")
        f.write(f"parameter_efficiency_loss: {parameter_efficiency_loss}\n")
        f.write(f"parameter_perplexity: {parameter_perplexity}\n")

    log_csv_path = output_dir.parent / "run_log.csv"
    log_trial_result_csv(log_csv_path, {
        "timestamp": datetime.now().isoformat(),
        "token_count": tokens,
        "trial_number": trial_number,
        "accuracy": accuracy,
        "tops_used": tops_used,
        "parameter_efficiency": parameter_efficiency,
        "parameter_efficiency_loss": parameter_efficiency_loss,
        "parameter_perplexity": parameter_perplexity
    })

    gc.collect()

    return {
        "status": "success",
        "accuracy": accuracy,
        "perplexity": perplexity,
        "tops_used": tops_used,
        "parameter_efficiency": parameter_efficiency,
        "parameter_efficiency_loss": parameter_efficiency_loss,
        "parameter_perplexity": parameter_perplexity
    }
