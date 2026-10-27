"""Sample 1,000 ImageNet validation images (seed 42) and save as PNG."""
import io
import os
import random
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

REPO = "benjamin-paine/imagenet-1k"
N_SHARDS, N_SAMPLES, SEED = 14, 1000, 42
OUT = Path(__file__).resolve().parents[1] / "data" / "imagenet_eval"
OUT.mkdir(parents=True, exist_ok=True)
tmp = "/tmp/imnet_val"  # fixed path so interrupted downloads resume
os.makedirs(tmp, exist_ok=True)

paths = [f"data/validation-{i:05d}-of-{N_SHARDS:05d}.parquet" for i in range(N_SHARDS)]
counts = []
for p in paths:
    f = hf_hub_download(REPO, p, repo_type="dataset", local_dir=tmp)
    counts.append(pq.ParquetFile(f).metadata.num_rows)
    print(p, counts[-1], flush=True)

total = sum(counts)
chosen = set(random.Random(SEED).sample(range(total), N_SAMPLES))

offset = 0
for p, c in zip(paths, counts):
    local = [i - offset for i in chosen if offset <= i < offset + c]
    if local:
        table = pq.read_table(os.path.join(tmp, p))
        for j in sorted(local):
            row = table.slice(j, 1).to_pylist()[0]
            img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
            img.save(OUT / f"{offset + j:05d}_label{row['label']}.png")
    offset += c
    os.remove(os.path.join(tmp, p))
print("saved", len(list(OUT.glob("*.png"))), "to", OUT)
