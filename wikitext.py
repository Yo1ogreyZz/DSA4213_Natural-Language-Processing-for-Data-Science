from datasets import load_dataset

ds = load_dataset("wikitext", "wikitext-2-v1")
with open("wikitext2_v1_full.txt", "w", encoding="utf-8") as f:
    for split in ["train", "validation", "test"]:
        for line in ds[split]["text"]:
            f.write(line + "\n")