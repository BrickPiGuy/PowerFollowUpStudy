'''More POWER: python -m pip install -U nvidia-ml-py
'''

from train_model import train_model
from pathlib import Path
import torch
import os

completed = 0
total = 50

tokens = 2_000_000
subset_path = "tinystories_subsets/tinystories_2M.jsonl"

for trial in range(1, total + 1):
    output_dir = Path(f"FUPowerResults/{tokens}_tokens/trial_{trial}")
    result_file = output_dir / "FUPowerResults.txt"

    if result_file.exists():
        print(f"✅ Skipping existing trial: {tokens} tokens, Trial {trial}")
        completed += 1
        continue

    print(f"🚀 Running trial {trial} for {tokens} tokens")
    try:
        result = train_model(
            tokens=tokens,
            trial_number=trial,
            output_dir=output_dir,
            s3_path=subset_path,
            verbose=False
        )
        torch.cuda.empty_cache()
        print(f"✅ Trial {trial} complete — ppl={result['perplexity']:.2f}, eff={result['parameter_efficiency']:.2e}")
        completed += 1
    except Exception as e:
        print(f"❌ Error during trial {trial} for {tokens} tokens: {e}")

print(f"🎉 Finished {completed} of {total} trials at 2M tokens.")