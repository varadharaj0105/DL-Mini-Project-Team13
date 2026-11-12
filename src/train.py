import argparse, copy, json, os, random, time
import numpy as np, pandas as pd, torch, torch.nn as nn, timm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

p = argparse.ArgumentParser()
p.add_argument("--root", required=True, help="folder containing split.csv and images")
p.add_argument("--model", default="mobilenetv3_large_100")
p.add_argument("--epochs", type=int, default=15)
p.add_argument("--bs", type=int, default=64)
p.add_argument("--lr", type=float, default=5e-4)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--variety_shift", action="store_true", help="paddy only: train ADT45, test other varieties")
p.add_argument("--out", default="/content/drive/MyDrive/DL_MiniProject/runs")
a = p.parse_args()

random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
cuda = torch.cuda.is_available()
dev = "cuda" if cuda else "cpu"
os.makedirs(a.out, exist_ok=True)
dname = os.path.basename(a.root.rstrip("/"))
tag = f"{dname}_{a.model}{'_vshift' if a.variety_shift else ''}_seed{a.seed}"

# ---- data ----
df = pd.read_csv(f"{a.root}/split.csv")
classes = sorted(df.label.unique()); cidx = {c: i for i, c in enumerate(classes)}
if a.variety_shift:
    tr = df[(df.variety == "ADT45") & (df.split == "train")]
    va = df[(df.variety == "ADT45") & (df.split == "val")]
    te = df[df.variety != "ADT45"]
else:
    tr, va, te = [df[df.split == s] for s in ("train", "val", "test")]
print(tag, "| train/val/test:", len(tr), len(va), len(te), "| classes:", len(classes))

norm = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
tf_tr = T.Compose([T.RandomResizedCrop(224, scale=(0.5, 1.0)), T.RandomHorizontalFlip(),
                   T.ColorJitter(0.3, 0.3, 0.3, 0.0), T.ToTensor(), norm])
tf_ev = T.Compose([T.Resize((224, 224)), T.ToTensor(), norm])

class DS(Dataset):
    def __init__(s, d, tf):
        s.p = d.path.tolist(); s.y = [cidx[l] for l in d.label]; s.tf = tf
    def __len__(s): return len(s.p)
    def __getitem__(s, i):
        im = Image.open(f"{a.root}/{s.p[i]}")
        im.draft("RGB", (448, 448))  # faster JPEG decode for big images
        return s.tf(im.convert("RGB")), s.y[i]

def loader(d, tf, shuffle):
    return DataLoader(DS(d, tf), batch_size=a.bs, shuffle=shuffle, num_workers=2,
                      pin_memory=cuda, drop_last=shuffle, persistent_workers=True, prefetch_factor=4)
trl, val, tel = loader(tr, tf_tr, True), loader(va, tf_ev, False), loader(te, tf_ev, False)

# ---- model / loss / optim ----
m = timm.create_model(a.model, pretrained=True, num_classes=len(classes)).to(dev)
cnt = tr.label.map(cidx).value_counts().sort_index().reindex(range(len(classes)), fill_value=1).values.astype(float)
w = (cnt.max() / cnt) ** 0.5; w = w / w.mean()   # softened inverse-frequency weights
crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=dev), label_smoothing=0.1)
opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.05)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(trl), pct_start=0.1)
scaler = torch.cuda.amp.GradScaler(enabled=cuda)

def logits(o):  # some timm models (e.g. EfficientFormer) return a tuple in train mode
    return (o[0] + o[1]) / 2 if isinstance(o, tuple) else o

@torch.no_grad()
def predict(ld):
    m.eval(); P, Y = [], []
    for x, y in ld:
        with torch.autocast(device_type=dev, dtype=torch.float16, enabled=cuda):
            o = logits(m(x.to(dev)))
        P += o.argmax(1).cpu().tolist(); Y += y.tolist()
    return np.array(Y), np.array(P)

# ---- train ----
best, ck = -1, f"{a.out}/{tag}.pt"
for ep in range(a.epochs):
    m.train(); t0 = time.time(); tl = 0
    for x, y in trl:
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=dev, dtype=torch.float16, enabled=cuda):
            loss = crit(logits(m(x)), y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        tl += loss.item()
    yv, pv = predict(val)
    f1 = f1_score(yv, pv, average="macro")
    print(f"ep {ep+1}/{a.epochs} loss {tl/len(trl):.4f} val_acc {accuracy_score(yv, pv):.4f} val_macroF1 {f1:.4f} ({time.time()-t0:.0f}s)", flush=True)
    if f1 > best:
        best = f1; torch.save(m.state_dict(), ck)

# ---- test with best checkpoint ----
m.load_state_dict(torch.load(ck, map_location=dev))
yt, pt = predict(tel)
acc, mf1 = accuracy_score(yt, pt), f1_score(yt, pt, average="macro")
print(classification_report(yt, pt, labels=range(len(classes)), target_names=classes, digits=4, zero_division=0))
pd.DataFrame(confusion_matrix(yt, pt, labels=range(len(classes))), index=classes, columns=classes).to_csv(f"{a.out}/{tag}_confusion.csv")

# ---- efficiency: params, GMACs, CPU latency (batch 1) ----
mc = copy.deepcopy(m).float().cpu().eval()
x1 = torch.randn(1, 3, 224, 224)
try:
    from torch.utils.flop_counter import FlopCounterMode
    with torch.no_grad(), FlopCounterMode(display=False) as fc: mc(x1)
    gmacs = fc.get_total_flops() / 2e9
except Exception as e:
    gmacs = None; print("flop count failed:", e)
with torch.no_grad():
    for _ in range(5): mc(x1)
    ts = []
    for _ in range(30):
        t0 = time.perf_counter(); mc(x1); ts.append((time.perf_counter() - t0) * 1000)
res = dict(tag=tag, test_acc=acc, test_macro_f1=mf1, best_val_macro_f1=best,
           per_class_f1=dict(zip(classes, f1_score(yt, pt, average=None, labels=range(len(classes)), zero_division=0).tolist())),
           params_M=sum(q.numel() for q in m.parameters()) / 1e6, gmacs=gmacs,
           cpu_latency_ms_median=float(np.median(ts)), n_train=len(tr), n_val=len(va), n_test=len(te), args=vars(a))
json.dump(res, open(f"{a.out}/{tag}.json", "w"), indent=1)
print(f"\nTEST acc {acc:.4f} macroF1 {mf1:.4f} | params {res['params_M']:.2f}M | GMACs {gmacs} | CPU latency {res['cpu_latency_ms_median']:.1f} ms")
