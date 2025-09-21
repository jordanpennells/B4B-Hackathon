# marbling_mvp.py
# Minimal AI Vision Grader backbone with robust labelling + audit
# Usage examples:
#   python marbling_mvp.py --dir images --check                      # audit labels only
#   python marbling_mvp.py --dir images --viz --limit 6 --drop-unknown
#   python marbling_mvp.py --images images\A.png images\B.png --viz
# Outputs: features.csv (indexed by path), labels_debug.csv (in --check)

import os, sys, glob, argparse
import fnmatch as fnm
import re
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

from skimage import color, exposure, measure
from skimage.morphology import footprint_rectangle, closing, remove_small_objects
from skimage.filters import threshold_otsu, gaussian
from skimage.feature import graycomatrix, graycoprops
from PIL import Image, ImageOps

# ----------------------------- Config (tweak here) -----------------------------
CFG = dict(
    min_obj_size=50,         # remove specks smaller than this (pixels)
    close_size=3,            # morphological closing kernel size
    invert_if_mean_gt=0.60,  # invert mask if >60% area (fail-safe)
    lacunarity_scales=(2,4,8,16,32),  # box sizes for lacunarity
)

# Optional alias keywords → grades (extend as you like)
# NOTE: Wagyu is now its own class, not mapped to Prime.
LABEL_ALIASES = {
    "wagyu": "Wagyu",
    # "angus": "Choice",     # uncomment if you want this behaviour
}
# ------------------------------------------------------------------------------

# ---------- Robust label inference ----------
def _norm_letters(s: str) -> str:
    return re.sub(r'[^a-z]+', '', s.lower())

PRIME_TOKENS  = {"abundant", "moderatelyabundant", "slightlyabundant"}
CHOICE_TOKENS = {"moderate", "modest", "small"}
SELECT_TOKENS = {"slight"}

def infer_label_from_name(name: str):
    raw = name.lower()
    norm = _norm_letters(name)

    # direct grade words
    if "wagyu" in raw:  return "Wagyu", "filename:wagyu"
    if "prime" in raw:  return "Prime", "filename:prime"
    if "choice" in raw: return "Choice", "filename:choice"
    if "select" in raw: return "Select", "filename:select"

    # alias keywords
    for k, v in LABEL_ALIASES.items():
        if k in raw:
            return v, f"filename:alias:{k}"

    # USDA degree mapping
    for t in PRIME_TOKENS:
        if t in norm: return "Prime", f"filename:{t}"
    for t in CHOICE_TOKENS:
        if t in norm: return "Choice", f"filename:{t}"
    for t in SELECT_TOKENS:
        if t in norm: return "Select", f"filename:{t}"

    # USDA prefixes (e.g., usdaprime)
    if "usdaprime"  in norm: return "Prime",  "filename:usdaprime"
    if "usdachoice" in norm: return "Choice", "filename:usdachoice"
    if "usdaselect" in norm: return "Select", "filename:usdaselect"

    return "unknown", "filename:none"

def infer_label(path: str):
    parent = os.path.basename(os.path.dirname(path)).lower()
    if parent in {"wagyu","prime","choice","select"}:
        return parent.title(), "folder"
    return infer_label_from_name(os.path.basename(path))

# ---------- IO helpers ----------
def load_rgb(path):
    """Robust image load → RGB (handles CMYK, odd ICC, EXIF orientation)."""
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    im = im.convert("RGB")
    return np.array(im)

# ---------- Pipeline ----------
def preprocess_rgb_to_gray(rgb):
    """CLAHE on Lab-L channel; return equalised gray [0,1] plus Lab a,b."""
    lab = color.rgb2lab(rgb)                  # L in [0..100]
    L, a, b = lab[...,0], lab[...,1], lab[...,2]
    Ln = (L/100.0).astype(np.float32)         # [0..1]
    L_eq = exposure.equalize_adapthist(Ln, clip_limit=0.02)
    return L_eq, a, b

def segment_fat(rgb, gray_eq, a_channel, cfg=CFG):
    """Segment fat as bright/less-red regions: score = gray_eq - α*norm(a) → Otsu + morph."""
    a_norm = (a_channel - np.min(a_channel)) / (np.ptp(a_channel) + 1e-7)
    score = gray_eq - 0.25 * a_norm
    t = threshold_otsu(score)
    mask = score > t
    mask = closing(mask, footprint_rectangle((cfg['close_size'], cfg['close_size'])))
    mask = remove_small_objects(mask, cfg['min_obj_size'])
    mask = measure.label(mask) > 0
    if mask.mean() > cfg['invert_if_mean_gt']:   # robustness to polarity
        mask = ~mask
        mask = closing(mask, footprint_rectangle((cfg['close_size'], cfg['close_size'])))
        mask = remove_small_objects(mask, cfg['min_obj_size'])
    return mask

def features_area_components(mask, props=None):
    if props is None:
        props = measure.regionprops(measure.label(mask))
    areas = [p.area for p in props]
    total = mask.size
    feats = {
        "area_pct": (100.0 * np.sum(areas) / total) if total else 0.0,
        "n_flecks": len(areas),
        "mean_fleck_area": float(np.mean(areas)) if areas else 0.0,
        "median_fleck_area": float(np.median(areas)) if areas else 0.0,
        "fleck_solidity": float(np.mean([p.solidity for p in props])) if areas else 0.0,
    }
    return feats

def features_glcm(gray_eq):
    """GLCM features (8 levels, 4 directions, distance 1 px)."""
    gray8 = np.clip((gray_eq * 7).round().astype(np.uint8), 0, 7)
    glcm = graycomatrix(gray8, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=8, symmetric=True, normed=True)
    feats = {}
    for feat in ['contrast', 'correlation', 'energy', 'homogeneity']:
        feats[f"glcm_{feat}"] = float(np.mean(graycoprops(glcm, feat)))
    return feats

def features_orientation_dispersion(mask):
    """Orientation dispersion via gradient angles on a smoothed mask."""
    m = gaussian(mask.astype(float), sigma=1.0)
    gy, gx = np.gradient(m)
    angles = np.arctan2(gy, gx)  # [-pi, pi]
    mag = np.hypot(gx, gy)
    mag = mag / (mag.max() + 1e-8)
    sel = mag > (0.2 * mag.mean())
    theta = angles[sel]; w = mag[sel]
    if theta.size == 0:
        return {"orientation_dispersion": 1.0, "orientation_anisotropy": 0.0}
    C = np.sum(w * np.cos(theta))
    S = np.sum(w * np.sin(theta))
    R = np.hypot(C, S) / (np.sum(w) + 1e-8)   # 0..1
    dispersion = 1.0 - R                      # 0 (aligned) .. 1 (isotropic)
    return {"orientation_dispersion": float(dispersion), "orientation_anisotropy": float(R)}

def features_lacunarity(mask, scales=CFG['lacunarity_scales']):
    """Lacunarity across box sizes; return mean and slope in log–log."""
    mask_u8 = mask.astype(np.uint8)
    Ls, lac = [], []
    for s in scales:
        k = np.ones((s, s), dtype=np.float32)
        summed = cv2.filter2D(mask_u8.astype(np.float32), -1, k, borderType=cv2.BORDER_REFLECT)
        mu = np.mean(summed); var = np.var(summed)
        cur = np.nan if mu <= 1e-8 else (1.0 + (var / (mu**2)))
        lac.append(cur); Ls.append(s)
    lac = np.array(lac, dtype=np.float64)
    valid = np.isfinite(lac)
    out = {
        "lacunarity_mean": float(np.nanmean(lac)),
        "lacunarity_slope": float(np.polyfit(np.log(np.array(Ls)[valid]), np.log(lac[valid] + 1e-9), 1)[0]) if valid.sum() >= 2 else 0.0
    }
    return out

def analyze_image(path):
    """Compute features for a single image path."""
    rgb = load_rgb(path)
    gray_eq, a_ch, _ = preprocess_rgb_to_gray(rgb)
    mask = segment_fat(rgb, gray_eq, a_ch, CFG)
    props = measure.regionprops(measure.label(mask))

    feats = {}
    feats.update(features_area_components(mask, props))
    feats.update(features_glcm(gray_eq))
    feats.update(features_orientation_dispersion(mask))
    feats.update(features_lacunarity(mask))
    return feats, rgb, gray_eq, mask

def run_on_images(image_paths, viz=False):
    rows = []; viz_pairs = []
    for p in image_paths:
        feats, rgb, g, m = analyze_image(p)
        feats['path']  = os.path.abspath(p)
        feats['image'] = os.path.basename(p)
        feats['label'] = os.path.basename(os.path.dirname(p))
        rows.append(feats)
        if viz:
            viz_pairs.append((os.path.basename(p), rgb, g, m, feats["area_pct"]))
    df = pd.DataFrame(rows).set_index("path")
    df.to_csv("features.csv")
    print(f"Saved features.csv with {len(df)} rows")
    if viz and viz_pairs:
        viz_pairs = viz_pairs[:6]  # cap previews
        n = len(viz_pairs)
        fig, ax = plt.subplots(n, 3, figsize=(10, 3*n))
        if n == 1: ax = np.array([ax])
        for i, (name, rgb, g, m, ap) in enumerate(viz_pairs):
            ax[i,0].imshow(rgb); ax[i,0].set_title(f"{name} (RGB)")
            ax[i,1].imshow(g, cmap='gray'); ax[i,1].set_title("Preprocessed (L-CLAHE)")
            ax[i,2].imshow(m, cmap='gray'); ax[i,2].set_title(f"Mask fat (area%={ap:.1f})")
            for j in range(3): ax[i,j].axis('off')
        plt.tight_layout(); plt.show()
    return df

def collect_image_paths(root_dir, exts=("*.png","*.jpg","*.jpeg","*.PNG","*.JPG","*.JPEG")):
    paths = []
    # under root
    for ext in exts:
        paths += glob.glob(os.path.join(root_dir, ext))
    # under subfolders
    for label_dir in sorted(glob.glob(os.path.join(root_dir, "*"))):
        if not os.path.isdir(label_dir):
            continue
        for ext in exts:
            paths += glob.glob(os.path.join(label_dir, ext))
    return sorted(paths)

def run_on_folder(root_dir, args, viz=False):
    image_paths = collect_image_paths(root_dir)
    if not image_paths:
        print(f"No images found in {root_dir}"); sys.exit(1)

    # pattern filter
    if args.pattern and args.pattern != "*":
        image_paths = [p for p in image_paths if fnm.fnmatch(os.path.basename(p), args.pattern)]
    if not image_paths:
        print("No images after pattern filter."); sys.exit(1)

    # infer labels (folder wins; else filename mapping)
    labeled = []
    for p in image_paths:
        label, src = infer_label(p)
        labeled.append((p, label, src))

    # optional drop unknowns
    if getattr(args, "drop_unknown", False):
        labeled = [t for t in labeled if t[1] != "unknown"]

    # optional label filter
    if args.label:
        labeled = [t for t in labeled if t[1] == args.label]
    if not labeled:
        print("No images to process after filters."); sys.exit(1)

    # shuffle/limit
    if args.shuffle:
        import random; random.shuffle(labeled)
    if args.limit and args.limit > 0:
        labeled = labeled[:args.limit]

    # --check mode: audit only
    if args.check:
        print("Audit of inferred labels (first 50 shown):")
        for i, (p, label, src) in enumerate(labeled[:50]):
            rel = os.path.relpath(p, root_dir)
            print(f"{rel} → {label} ({src})")
        dbg = pd.DataFrame([{
            "path": os.path.abspath(p),
            "image": os.path.basename(p),
            "relpath": os.path.relpath(p, root_dir),
            "label": label,
            "source": src
        } for p, label, src in labeled])
        dbg.to_csv("labels_debug.csv", index=False)
        print("Wrote labels_debug.csv.")
        print("Label counts:\n", dbg["label"].value_counts())
        return None

    # compute features
    rows = []; viz_pairs = []
    for p, label, src in labeled:
        feats, rgb, g, m = analyze_image(p)
        feats["path"]  = os.path.abspath(p)
        feats["image"] = os.path.basename(p)
        feats["relpath"] = os.path.relpath(p, root_dir)
        feats["label"] = label
        feats["label_source"] = src
        rows.append(feats)
        if viz:
            viz_pairs.append((os.path.basename(p), rgb, g, m, feats["area_pct"]))

    df = pd.DataFrame(rows).set_index("path")
    df.to_csv("features.csv")
    print(f"Saved features.csv with {len(df)} rows (labels inferred from folder/filename)")
    print("Label counts:\n", df["label"].value_counts())

    if viz and viz_pairs:
        viz_pairs = viz_pairs[:6]  # cap previews
        n = len(viz_pairs)
        fig, ax = plt.subplots(n, 3, figsize=(10, 3*n))
        if n == 1: ax = np.array([ax])
        for i, (name, rgb, g, m, ap) in enumerate(viz_pairs):
            ax[i,0].imshow(rgb); ax[i,0].set_title(f"{name} (RGB)")
            ax[i,1].imshow(g, cmap='gray'); ax[i,1].set_title("Preprocessed (L-CLAHE)")
            ax[i,2].imshow(m, cmap='gray'); ax[i,2].set_title(f"Mask fat (area%={ap:.1f})")
            for j in range(3): ax[i,j].axis('off')
        plt.tight_layout(); plt.show()
    return df

def main():
    ap = argparse.ArgumentParser(description="Minimal marbling feature extractor (MVP)")
    ap.add_argument("--images", nargs="+", help="List of image paths")
    ap.add_argument("--dir", type=str, help="Folder with images and/or subfolders")
    ap.add_argument("--viz", action="store_true", help="Show quick segmentation figures (caps at 6)")
    ap.add_argument("--label", type=str, help="Only process this label (Wagyu/Prime/Choice/Select)")
    ap.add_argument("--pattern", type=str, default="*", help="Filename glob, e.g. *Wagyu*.png")
    ap.add_argument("--limit", type=int, default=0, help="Limit total images (0 = all)")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle before limiting")
    ap.add_argument("--check", action="store_true", help="Audit inferred labels only (no feature extraction)")
    ap.add_argument("--drop-unknown", action="store_true", help="Skip images we cannot confidently label")
    args = ap.parse_args()

    if args.images:
        df = run_on_images(args.images, viz=args.viz)
        if len(args.images) == 2:
            i1, i2 = [os.path.abspath(p) for p in args.images]
            sub = df.loc[[i1, i2]].T.reset_index()
            sub.columns = ["Feature", os.path.basename(i1), os.path.basename(i2)]
            sub["Difference"] = sub.iloc[:,1] - sub.iloc[:,2]
            sub.to_csv("marbling_feature_comparison.csv", index=False)
            print("Saved marbling_feature_comparison.csv")
    elif args.dir:
        run_on_folder(args.dir, args, viz=args.viz)
    else:
        print("Provide --images path1 path2 ... OR --dir images"); sys.exit(1)

if __name__ == "__main__":
    main()
