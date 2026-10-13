import torch
import json
import sys
from pathlib import Path

checkpoint_path = sys.argv[1]

print("Loading checkpoint metadata and tensors...")

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    mmap=True,
    weights_only=False,
)

state_dict = checkpoint.get("state_dict", checkpoint)

results = {}

for name, tensor in state_dict.items():

    if not isinstance(tensor, torch.Tensor):
        continue

    if (
        "sampling_offsets" not in name
        and "attention_weights" not in name
    ):
        continue

    data = tensor.detach().float()

    results[name] = {
        "shape": list(data.shape),
        "mean": data.mean().item(),
        "std": data.std(unbiased=False).item(),
        "min": data.min().item(),
        "max": data.max().item(),
    }

    print(f"\n{name}")
    print(results[name])

output_path = Path("work_dirs/sampling_offsets_summary.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved: {output_path}")
