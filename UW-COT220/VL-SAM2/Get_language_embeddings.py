import json
import hashlib
from pathlib import Path

import clip
import torch
from tqdm import tqdm
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main():
    device = "cuda"

    root_path = Path("/mnt/sdb/zhangchunhui/Datasets/YouTube-VOS/meta_expressions")
    save_dir = Path("/mnt/sdb/zhangchunhui/SAM2/VL-SAM2/cache/clip_lang")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load the dataset
    all_texts = []
    for json_path in root_path.rglob("meta_expressions.json"):
        with open(json_path) as f:
            annos = json.load(f)["videos"]

        for _, anno in annos.items():
            expressions = anno["expressions"]
            for _, exp in expressions.items():
                exp = exp["exp"]
                all_texts.append(exp)

    all_texts.append("Object")

    all_texts = sorted(list(set(all_texts)))
    print(f"Number of unique expressions: {len(all_texts)}")

    # Load the model
    model, _ = clip.load("ViT-B/32", device=device, jit=False)

    # Precompute the embeddings
    batch_size = 64
    for i in tqdm(range(0, len(all_texts), batch_size)):
        texts = all_texts[i : i + batch_size]
        input_ids = clip.tokenize(texts, truncate=True).to(device)

        with torch.no_grad():
            text_features = model.encode_text(input_ids)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.cpu()

        for text, text_feature in zip(texts, text_features):
            m = hashlib.sha256()
            m.update(text.encode("utf-8"))
            text_hash = m.hexdigest()
            torch.save(text_feature, f"/mnt/sdb/zhangchunhui/SAM2/VL-SAM2/cache/clip_lang/{text_hash}.pt")
'''
cd /mnt/sdb/zhangchunhui/SAM2/VL-SAM2/
conda activate SAM2 
python Get_language_embeddings.py   
'''
if __name__ == "__main__":
    main()
