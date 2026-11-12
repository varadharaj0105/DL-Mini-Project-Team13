"""Download both datasets and build leakage-aware split files.

Usage:  python src/prepare_data.py paddy  --root data/paddy
        python src/prepare_data.py plantdoc --root data/plantdoc
Creates <root>/split.csv (columns: path,label,split; Paddy also variety,age,group).
"""
import argparse, glob, os, subprocess
import numpy as np, pandas as pd


def paddy(root):
    from datasets import load_dataset
    import imagehash
    from PIL import Image
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from sklearn.model_selection import StratifiedGroupKFold

    ds = load_dataset("PawanKapakayala/paddy-disease-classification", split="train")  # 10,407 images, 10 classes
    lab, var = ds.features["label"].names, ds.features["variety"].names
    for n in lab: os.makedirs(f"{root}/{n}", exist_ok=True)
    rows = []
    for i, ex in enumerate(ds):
        rel = f"{lab[ex['label']]}/{i:05d}.jpg"
        ex["image"].convert("RGB").save(f"{root}/{rel}", quality=95)
        rows.append((rel, lab[ex["label"]], var[ex["variety"]], ex["age"]))
    df = pd.DataFrame(rows, columns=["path", "label", "variety", "age"])

    # near-duplicate groups: pHash, Hamming distance <= 4, connected components
    H = np.array([imagehash.phash(Image.open(f"{root}/{p}")).hash.flatten() for p in df.path]).astype(np.float32)
    r_, c_ = [], []
    for s in range(0, len(H), 1000):
        d = 64 - (H[s:s + 1000] @ H.T + (1 - H[s:s + 1000]) @ (1 - H).T)
        r, c = np.where(d <= 4); r_ += list(r + s); c_ += list(c)
    g = coo_matrix((np.ones(len(r_)), (r_, c_)), shape=(len(H), len(H)))
    n, df["group"] = connected_components(g, directed=False)
    sizes = df.groupby("group").path.transform("count")
    print("groups:", n, "| images in near-dup groups:", (sizes > 1).sum(), "| largest group:", sizes.max())

    # stratified group split: ~1/7 test, then ~1/6 of the rest validation
    trv, te = next(StratifiedGroupKFold(7, shuffle=True, random_state=42).split(df, df.label, df.group))
    rest = df.iloc[trv]
    _, va = next(StratifiedGroupKFold(6, shuffle=True, random_state=42).split(rest, rest.label, rest.group))
    df["split"] = "train"; df.loc[te, "split"] = "test"; df.loc[rest.index[va], "split"] = "val"
    df.to_csv(f"{root}/split.csv", index=False); print(df.split.value_counts())


def plantdoc(root):
    from sklearn.model_selection import train_test_split
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/pratikkayal/PlantDoc-Dataset.git", root], check=True)
    rows = []
    for split in ("train", "test"):
        for c in sorted(os.listdir(f"{root}/{split}")):
            for f in glob.glob(f"{root}/{split}/{c}/*"):
                rows.append((os.path.relpath(f, root), c, split))
    pdf = pd.DataFrame(rows, columns=["path", "label", "orig_split"])
    pdf = pdf[pdf.label != "Tomato two spotted spider mites leaf"].reset_index(drop=True)  # no test images -> 27 classes
    tr = pdf[pdf.orig_split == "train"]
    t_idx, v_idx = train_test_split(tr.index, test_size=0.15, stratify=tr.label, random_state=42)
    pdf["split"] = "test"; pdf.loc[t_idx, "split"] = "train"; pdf.loc[v_idx, "split"] = "val"
    pdf.to_csv(f"{root}/split.csv", index=False); print(pdf.split.value_counts())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("dataset", choices=["paddy", "plantdoc"]); ap.add_argument("--root", required=True)
    a = ap.parse_args(); {"paddy": paddy, "plantdoc": plantdoc}[a.dataset](a.root)
