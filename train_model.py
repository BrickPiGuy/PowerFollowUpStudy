import os
import gc
import csv
import json
import math
import time
import boto3
import torch
import tempfile
import threading
from typing import Optional
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.amp import autocast, GradScaler

# (Optional) quiet TensorFlow/JAX noise if they exist in the env
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Benchmark results (pick the greater) ---
FP16_TFLOPS = 67.34
FP16_TIME_S = 0.82
BF16_TFLOPS = 68.16
BF16_TIME_S = 0.81

CS_TFLOPS = BF16_TFLOPS if BF16_TFLOPS >= FP16_TFLOPS else FP16_TFLOPS  # -> 68.16
CS_MODE = "BF16" if BF16_TFLOPS >= FP16_TFLOPS else "FP16"


# -----------------------------
# Utilities
# -----------------------------
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rms(values):
    """Root-mean-square for a sequence of numbers."""
    if not values:
        return float("nan")
    return math.sqrt(sum(v * v for v in values) / len(values))


def compute_pe(inv_perplexity, cs_tflops, watts_series, ms_params, tt_tokens,
               tt_scale=1_000_000, K=1.0):
    """
    New Parameter Efficiency (PE):
        PE = K * (inv_perplexity * (CS_TFLOPS / RMS(watts))) / (MS_params * TT_norm)
    Returns float('nan') if inputs are insufficient.
    """
    if not (isinstance(inv_perplexity, (int, float)) and inv_perplexity >= 0):
        return float("nan")
    if not (isinstance(cs_tflops, (int, float)) and cs_tflops > 0):
        return float("nan")
    if not watts_series:
        return float("nan")
    watts_rms = rms(watts_series)
    if not (isinstance(watts_rms, (int, float)) and watts_rms > 0):
        return float("nan")
    if not (isinstance(ms_params, (int, float)) and ms_params > 0):
        return float("nan")
    if not (isinstance(tt_tokens, (int, float)) and tt_tokens > 0):
        return float("nan")

    ppw = cs_tflops / watts_rms                   # TFLOPS per watt
    tt_norm = tt_tokens / tt_scale                # tokens in "millions" if tt_scale=1e6
    return K * (inv_perplexity * ppw) / (ms_params * tt_norm)


# -----------------------------
# Model & Tokenizer
# -----------------------------
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


# -----------------------------
# Dataset
# -----------------------------
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


# -----------------------------
# Optimizer
# -----------------------------
def get_optimizer(model, learning_rate=1e-6):
    return AdamW(model.parameters(), lr=learning_rate)


# -----------------------------
# Evaluation
# -----------------------------
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


# -----------------------------
# CSV logging (tops removed; extra metrics added)
# -----------------------------
def log_trial_result_csv(csv_path: Path, trial_data: dict):
    """
    Keeps the legacy 'tops_used' column for compatibility; writes an empty value.
    Adds epochs, power_samples (count), and watts_rms to each row.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp", "token_count", "trial_number",
        "accuracy", "tops_used",  # deprecated; kept empty
        "parameter_efficiency", "parameter_efficiency_loss", "parameter_perplexity",
        "epochs", "power_samples", "watts_rms"
    ]
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        # Ensure the deprecated column exists with an empty value
        trial_data = {**trial_data, "tops_used": ""}
        writer.writerow(trial_data)


# -----------------------------
# Background power sampler (Option B)
# -----------------------------
class PowerSampler:
    """Samples GPU power via NVML every `interval_s` seconds in a background thread."""
    def __init__(self, nvml, handle, interval_s: float = 60.0):
        self.nvml = nvml
        self.handle = handle
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thr = None
        self.samples = []

    def _run(self):
        # Immediate first sample
        try:
            p_mw = self.nvml.nvmlDeviceGetPowerUsage(self.handle)
            self.samples.append(p_mw / 1000.0)  # -> watts
        except Exception:
            pass
        # Fixed-interval loop
        while not self._stop.wait(self.interval_s):
            try:
                p_mw = self.nvml.nvmlDeviceGetPowerUsage(self.handle)
                self.samples.append(p_mw / 1000.0)
            except Exception:
                pass

    def start(self):
        self._thr = threading.Thread(target=self._run, daemon=True)
        self._thr.start()

    def stop(self):
        self._stop.set()
        if self._thr is not None:
            self._thr.join()
        return list(self.samples)  # return a copy


# -----------------------------
# NVML handle helper (robust fallback)
# -----------------------------
def get_nvml_handle_any(nvml):
    """
    Try to get a working NVML handle:
    1) index 0
    2) otherwise, first device that returns a valid power reading
    """
    try:
        h = nvml.nvmlDeviceGetHandleByIndex(0)
        # sanity-check a read
        _ = nvml.nvmlDeviceGetPowerUsage(h)
        return h
    except Exception:
        pass

    count = nvml.nvmlDeviceGetCount()
    for i in range(count):
        try:
            h = nvml.nvmlDeviceGetHandleByIndex(i)
            _ = nvml.nvmlDeviceGetPowerUsage(h)
            return h
        except Exception:
            continue
    raise RuntimeError("No NVML device returned a valid power reading.")


# -----------------------------
# Training + PE integration (no TOPS, Option B sampler)
# -----------------------------
def train_model(tokens: int,
                trial_number: int,
                output_dir: Path,
                s3_path: str,
                verbose: bool = True,
                cs_tflops: Optional[float] = CS_TFLOPS,  # default to the greater benchmark (68.16, BF16)
                epochs: int = 3,
                power_sample_every_s: float = 60.0) -> dict:
    """
    Train briefly and compute metrics.
    Args:
        tokens: target number of *true* tokens to include in the dataset.
        cs_tflops: measured compute speed in TFLOPS (from your benchmark). Defaults to best of FP16/BF16.
        epochs: number of passes over the dataset (affects tokens processed).
        power_sample_every_s: cadence for NVML power samples (background thread).
    """
    device = get_device()
    model = get_model(device=device)
    tokenizer = get_tokenizer()
    dataloader = get_dataloader(tokens, s3_path=s3_path)
    optimizer = get_optimizer(model)

    model.train()
    valid_batches = 0

    # Start background power sampler if NVML is available
    power_samples = []
    sampler = None
    nvml = None  # for shutdown later
    if device.type == "cuda":
        try:
            import pynvml as _nvml
            _nvml.nvmlInit()

            # try to find any valid device handle (index 0 preferred)
            handle = get_nvml_handle_any(_nvml)

            # take one synchronous sample right now (guarantees at least 1)
            try:
                p_mw = _nvml.nvmlDeviceGetPowerUsage(handle)
                power_samples.append(p_mw / 1000.0)
            except Exception as e:
                if verbose:
                    print(f"⚠️ Initial NVML power read failed: {e}")

            # start background sampler
            sampler = PowerSampler(_nvml, handle, interval_s=power_sample_every_s)
            sampler.start()
            nvml = _nvml
            if verbose:
                print(f"⚡ Power sampling every {power_sample_every_s:.0f}s (background thread).")
        except Exception as e:
            sampler = None
            if verbose:
                print(f"ℹ️ NVML not available or handle resolution failed: {e}. Power sampling disabled.")

    scaler = GradScaler(enabled=(device.type == "cuda"))

    # Optional: print which TFLOPS mode we're using
    mode_label = CS_MODE if (cs_tflops == CS_TFLOPS) else "custom"
    if verbose:
        print(f"⚙️ Using cs_tflops={cs_tflops} ({mode_label})")

    try:
        for epoch in range(epochs):
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
    finally:
        # Stop the sampler and shut down NVML
        if sampler is not None:
            try:
                power_samples = sampler.stop()
            except Exception:
                power_samples = []
        if nvml is not None:
            try:
                nvml.nvmlShutdown()
            except Exception:
                pass

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

    # --- Compute PE with the NEW formula (no TOPS anywhere) ---
    ms_params = float(sum(p.numel() for p in model.parameters()))
    tokens_processed = float(tokens * epochs)

    # Additional logging metrics
    watts_rms = rms(power_samples)
    samples_count = len(power_samples)

    parameter_efficiency = compute_pe(
        inv_perplexity=inv_perplexity,
        cs_tflops=cs_tflops if cs_tflops is not None else float("nan"),
        watts_series=power_samples,
        ms_params=ms_params,
        tt_tokens=tokens_processed,
        tt_scale=1_000_000,
        K=1.0
    )

    # Optional normalized perplexity per parameter
    parameter_perplexity = (round(perplexity / ms_params, 20)
                            if math.isfinite(perplexity) else float("nan"))

    # Relative to a baseline PE (kept from your prior code)
    if math.isfinite(parameter_efficiency):
        baseline_parameter_efficiency = 5e-9
        parameter_efficiency_loss = round(1 - (parameter_efficiency / baseline_parameter_efficiency), 10)
    else:
        parameter_efficiency_loss = float("nan")

    if verbose:
        print("✅ Training complete — preparing to log results...")

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.txt", "w") as f:
        f.write(f"perplexity: {perplexity}\n")
        f.write(f"inv_perplexity: {inv_perplexity}\n")
        f.write(f"cs_tflops: {cs_tflops} ({mode_label})\n")
        f.write(f"epochs: {epochs}\n")
        f.write(f"power_samples: {samples_count}\n")
        f.write(f"watts_rms: {watts_rms}\n")
        f.write(f"parameter_efficiency: {parameter_efficiency}\n")
        f.write(f"parameter_efficiency_loss: {parameter_efficiency_loss}\n")
        f.write(f"parameter_perplexity: {parameter_perplexity}\n")

    # CSV: write row with extra metrics
    log_csv_path = output_dir.parent / "run_log.csv"
    log_trial_result_csv(log_csv_path, {
        "timestamp": datetime.now().isoformat(),
        "token_count": tokens,
        "trial_number": trial_number,
        "accuracy": inv_perplexity,  # legacy alias for inv_perplexity
        "parameter_efficiency": parameter_efficiency,
        "parameter_efficiency_loss": parameter_efficiency_loss,
        "parameter_perplexity": parameter_perplexity,
        "epochs": epochs,
        "power_samples": samples_count,
        "watts_rms": watts_rms
    })

    # Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "status": "success",
        "inv_perplexity": inv_perplexity,     # explicit metric name
        "accuracy": inv_perplexity,           # legacy alias for compatibility
        "perplexity": perplexity,
        "parameter_efficiency": parameter_efficiency,            # NEW formula
        "parameter_efficiency_loss": parameter_efficiency_loss,  # vs. baseline
        "parameter_perplexity": parameter_perplexity,
        "epochs": epochs,
        "power_samples": samples_count,
        "watts_rms": watts_rms
    }