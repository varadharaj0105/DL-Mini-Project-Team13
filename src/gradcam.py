"""Grad-CAM of baseline vs distilled student on PlantDoc test images the baseline gets wrong and the distilled model gets right."""
import argparse, timm, torch, pandas as pd, matplotlib.pyplot as plt, torchvision.transforms as T
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/content/plantdoc_384"); ap.add_argument("--runs", required=True)
ap.add_argument("--base_tag", default="plantdoc_384_mobilenetv3_large_100_seed42")
ap.add_argument("--kd_tag", default="plantdoc_384_mobilenetv3_large_100_faug_kd_seed42")
ap.add_argument("--out", default="gradcam_plantdoc.png"); ap.add_argument("--n", type=int, default=6)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
df = pd.read_csv(f"{a.root}/split.csv"); classes = sorted(df.label.unique()); K = len(classes); cidx = {c: i for i, c in enumerate(classes)}
def load(tag):
    m = timm.create_model("mobilenetv3_large_100", pretrained=False, num_classes=K)
    m.load_state_dict(torch.load(f"{a.runs}/{tag}.pt", map_location="cpu")); return m.eval().to(dev)
base, kd = load(a.base_tag), load(a.kd_tag)
norm = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)); tf = T.Compose([T.Resize((224, 224)), T.ToTensor()])
def pred(m, x):
    with torch.no_grad(): return m(norm(x)[None].to(dev)).argmax(1).item()
rows = []
for _, r in df[df.split == "test"].reset_index(drop=True).iterrows():
    x = tf(Image.open(f"{a.root}/{r.path}").convert("RGB")); y = cidx[r.label]
    if pred(base, x) != y and pred(kd, x) == y: rows.append((x, y))
    if len(rows) == a.n: break
cams = [None, GradCAM(model=base, target_layers=[base.blocks[-1]]), GradCAM(model=kd, target_layers=[kd.blocks[-1]])]
fig, ax = plt.subplots(len(rows), 3, figsize=(9, 3 * len(rows)))
for j, (x, y) in enumerate(rows):
    rgb = x.permute(1, 2, 0).numpy()
    for k, cam in enumerate(cams):
        if cam is None: ax[j, k].imshow(rgb); ax[j, k].set_title(classes[y], fontsize=8)
        else:
            g = cam(input_tensor=norm(x)[None].to(dev), targets=[ClassifierOutputTarget(y)])[0]
            ax[j, k].imshow(show_cam_on_image(rgb, g, use_rgb=True))
            if j == 0: ax[j, k].set_title(["", "baseline", "distilled"][k])
        ax[j, k].axis("off")
plt.tight_layout(); plt.savefig(a.out, dpi=150)
