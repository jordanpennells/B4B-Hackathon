import os
import math
import joblib
import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Reuse the exact feature pipeline from your MVP (ensures parity with training)
import marbling_mvp as mvp
import skimage.segmentation as seg  # for optional edge overlay
from streamlit.components.v1 import html as st_html

st.set_page_config(
    page_title="AI Marbling Grader",
    layout="wide",
    initial_sidebar_state="collapsed",  # demo-friendly
)

# -------------------- Feature glossary (hover tooltips) --------------------
FEATURE_DEFS = {
    "area_pct": "Percentage of image area classified as fat (mask coverage). Higher → richer marbling.",
    "n_flecks": "Count of distinct fat regions (connected components). Many small flecks → fine marbling.",
    "orientation_anisotropy": "Directional alignment of fat structures (0–1). Higher → more aligned strands; lower → isotropic flecks.",
    "lacunarity_mean": "Measure of gaps/texture heterogeneity across scales. Higher → more irregular, ‘holey’ texture.",
    "glcm_contrast": "Texture contrast from Grey-Level Co-occurrence Matrix. Higher → stronger intensity differences between neighbouring pixels.",
    "meat_pct": "100 − area_pct. Proxy for lean proportion.",
}

def render_feature_table_html(features: dict) -> str:
    """Return a themed HTML table with hover tooltips on feature names."""
    rows_html = []
    for name, value in features.items():
        desc = FEATURE_DEFS.get(name, "No description available.")
        v = "nan" if value is None or not np.isfinite(value) else f"{float(value):.3f}"
        rows_html.append(
            f"""
            <tr class="row">
              <td class="name"><span title="{desc}">{name}</span></td>
              <td class="val">{v}</td>
            </tr>
            """
        )

    css = """
    <style>
    .feat-wrap { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }
    /* Theme tokens (dark default) */
    .feat-wrap {
      --bg: #0e1117; --bg-alt: #111827; --row: #0f172a; --row-alt: #0b1220;
      --border: #334155; --text: #e5e7eb; --muted: #9ca3af; --accent: #22c55e;
    }
    @media (prefers-color-scheme: light) {
      .feat-wrap {
        --bg: #ffffff; --bg-alt: #f8fafc; --row: #ffffff; --row-alt: #f9fafb;
        --border: #e5e7eb; --text: #111827; --muted: #6b7280; --accent: #16a34a;
      }
    }
    .feat-card { background: var(--bg-alt); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }
    .feat-table { width: 100%; border-collapse: collapse; color: var(--text); font-size: 0.95rem; }
    .feat-table th { text-align: left; padding: 10px 12px; background: var(--bg); border-bottom: 1px solid var(--border); font-weight: 700; }
    .feat-table td { padding: 8px 12px; border-bottom: 1px solid var(--border); }
    .feat-table td.name span[title] { cursor: help; border-bottom: 1px dotted var(--muted); }
    .feat-table td.val { text-align: right; font-variant-numeric: tabular-nums; }
    .feat-table tr.row:nth-child(odd) td { background: var(--row); }
    .feat-table tr.row:nth-child(even) td { background: var(--row-alt); }
    .feat-table tr.row:hover td.name span { color: var(--accent); }
    </style>
    """

    html = f"""
    <div class="feat-wrap">
      {css}
      <div class="feat-card">
        <table class="feat-table">
          <thead><tr><th>Feature</th><th>Value</th></tr></thead>
          <tbody>{''.join(rows_html)}</tbody>
        </table>
      </div>
    </div>
    """
    return html

# -------------------- Helpers --------------------
@st.cache_data
def load_csv(path):
    df = pd.read_csv(path)
    for c in ["prob_Select", "prob_Choice", "prob_Prime"]:
        if c not in df.columns:
            df[c] = np.nan
    if "confidence" not in df.columns:
        prob_cols = [c for c in df.columns if c.startswith("prob_")]
        df["confidence"] = df[prob_cols].max(axis=1) if prob_cols else np.nan
    if "prime_similarity" not in df.columns and "prob_Prime" in df.columns:
        df["prime_similarity"] = df["prob_Prime"]
    if "meat_pct" not in df.columns and "area_pct" in df.columns:
        df["meat_pct"] = 100 - df["area_pct"]
    if "image" not in df.columns and "path" in df.columns:
        df["image"] = df["path"].apply(os.path.basename)
    return df

def img_path(row, images_root):
    for key in ("path", "relpath"):
        if key in row and isinstance(row[key], str) and os.path.exists(row[key]):
            return row[key]
        if key in row and isinstance(row[key], str):
            candidate = os.path.join(images_root, row[key])
            if os.path.exists(candidate):
                return candidate
    if "image" in row and isinstance(row["image"], str):
        candidate = os.path.join(images_root, row["image"])
        if os.path.exists(candidate):
            return candidate
    return None

def badge_colour(conf):
    if pd.isna(conf): return "🔘"
    if conf >= 0.8:   return "🟢"
    if conf >= 0.6:   return "🟠"
    return "🔴"

def safe_img(path, max_w=512):
    try:
        im = Image.open(path).convert("RGB")
        w, h = im.size
        if w > max_w:
            im = im.resize((max_w, int(h * max_w / w)))
        return im
    except Exception:
        return None

def segment_fat_with_alpha(gray_eq, a_channel, alpha=0.25, cfg=mvp.CFG):
    """Same as MVP segmentation but with tunable alpha on the Lab-a term."""
    a_norm = (a_channel - np.min(a_channel)) / (np.ptp(a_channel) + 1e-7)
    score = gray_eq - float(alpha) * a_norm
    t = mvp.threshold_otsu(score)
    mask = score > t
    mask = mvp.closing(mask, mvp.footprint_rectangle((cfg['close_size'], cfg['close_size'])))
    mask = mvp.remove_small_objects(mask, cfg['min_obj_size'])
    mask = mvp.measure.label(mask) > 0
    if mask.mean() > cfg['invert_if_mean_gt']:
        mask = ~mask
        mask = mvp.closing(mask, mvp.footprint_rectangle((cfg['close_size'], cfg['close_size'])))
        mask = mvp.remove_small_objects(mask, cfg['min_obj_size'])
    return mask

@st.cache_resource
def load_model_bundle(path="grader_model.pkl"):
    try:
        return joblib.load(path)
    except Exception as e:
        st.warning(f"Could not load model bundle ({path}): {e}")
        return None

# -------------------- Sidebar (Settings only) --------------------
st.sidebar.title("Settings")
csv_path = st.sidebar.text_input(
    "Predictions CSV",
    value="predictions_dashboard.csv",
    help="Merged CSV from prep_dashboard.py",
)
images_root = st.sidebar.text_input(
    "Images root folder",
    value="images",
    help="Base path used to resolve image thumbnails.",
)
st.sidebar.caption("Open **Gallery & QA** to access filters.")

bundle = load_model_bundle()

if not os.path.exists(csv_path):
    st.warning(f"CSV not found: {csv_path}")
    st.stop()

df = load_csv(csv_path)

# -------------------- Title & Persistent View Switch --------------------
st.title("AI Marbling Grader – Dashboard")

# Default to Guided Walkthrough on first load
if "active_view" not in st.session_state:
    st.session_state.active_view = "🧭 Guided Walkthrough"

view_options = ["🧭 Guided Walkthrough", "🖼️ Gallery & QA"]

view = st.radio(
    "View",
    view_options,
    index=view_options.index(st.session_state.active_view),
    horizontal=True,
    key="active_view",
    help="Choose the dashboard view. This selection persists when the app reruns."
)

# ==================== GALLERY VIEW ====================
if view == "🖼️ Gallery & QA":
    cols = st.columns(4)
    with cols[0]:
        st.metric("Rows", len(df))
    with cols[1]:
        if "predicted" in df.columns:
            st.metric("Pred: Prime %", f"{(df['predicted']=='Prime').mean()*100:.1f}%")
    with cols[2]:
        if "area_pct" in df.columns:
            st.metric("Median fat %", f"{df['area_pct'].median():.1f}")
    with cols[3]:
        st.metric("High-confidence (≥0.8)", f"{(df['confidence']>=0.8).mean()*100:.1f}%")

    # --- On-page Filters (collapsed by default) ---
    with st.expander("Filters", expanded=False):
        pred_opts = sorted(df["predicted"].dropna().unique().tolist()) if "predicted" in df.columns else []
        true_opts = sorted(df["label"].dropna().unique().tolist()) if "label" in df.columns else []

        c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.0, 1.0])
        with c1:
            pred_sel = st.multiselect("Predicted", pred_opts, default=pred_opts,
                                      help="Show only these predicted grades.")
        with c2:
            true_sel = st.multiselect("True (if present)", true_opts, default=true_opts,
                                      help="Restrict to these ground-truth labels.")
        with c3:
            sort_by = st.selectbox("Sort by", ["prime_similarity","confidence","area_pct","meat_pct"])
        with c4:
            ascending = st.checkbox("Ascending", value=False)

        conf_rng = st.slider("Confidence range", 0.0, 1.0, (0.0, 1.0), 0.01)

        c5, c6 = st.columns(2)
        with c5:
            page_size = st.selectbox("Items per page", [6, 9, 12, 15, 20], index=1)
        with c6:
            page = st.number_input("Page", min_value=1, value=1, step=1)

    # Confusion (if ground truth present)
    if "label" in df.columns and "predicted" in df.columns:
        labels = ["Select", "Choice", "Prime"]
        cm = confusion_matrix(df["label"], df["predicted"], labels=labels)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(cm, interpolation="nearest")
        ax.set_title("Confusion")
        ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels); ax.set_yticklabels(labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center")
        fig.tight_layout()
        st.pyplot(fig, width="stretch")

    st.divider()

    # Apply Filter + sort
    work = df.copy()
    if "predicted" in work.columns and pred_sel:
        work = work[work["predicted"].isin(pred_sel)]
    if "label" in work.columns and ("true_sel" in locals()) and true_sel:
        work = work[work["label"].isin(true_sel)]
    work = work[(work["confidence"] >= conf_rng[0]) & (work["confidence"] <= conf_rng[1])]
    if ("sort_by" in locals()) and sort_by in work.columns:
        work = work.sort_values(sort_by, ascending=ascending)

    # Cards grid
    start = (page - 1) * page_size
    end = start + page_size
    subset = work.iloc[start:end]

    st.write(f"Showing **{len(subset)}** of {len(work)} filtered rows (page {page})")

    n_cols = 3
    rows = math.ceil(max(1, len(subset)) / n_cols)
    for r in range(rows):
        cols = st.columns(n_cols)
        for c in range(n_cols):
            idx = r * n_cols + c
            if idx >= len(subset): continue
            row = subset.iloc[idx].to_dict()
            imgfile = img_path(row, images_root)
            with cols[c]:
                st.markdown(f"### {row.get('image','(image)')}")
                if imgfile and os.path.exists(imgfile):
                    im = safe_img(imgfile, max_w=480)
                    if im is not None:
                        st.image(im, width="stretch")
                meta = []
                if "predicted" in row: meta.append(f"**Pred:** {row['predicted']}")
                if "label" in row:     meta.append(f"**Truth:** {row['label']}")
                meta.append(f"{badge_colour(row.get('confidence'))} **Conf:** {row.get('confidence', np.nan):.2f}")
                if "area_pct" in row:  meta.append(f"**Fat %:** {row['area_pct']:.1f}")
                if "meat_pct" in row:  meta.append(f"**Meat %:** {row['meat_pct']:.1f}")
                if "prime_similarity" in row: meta.append(f"**P(Prime):** {row['prime_similarity']:.2f}")
                st.markdown(" • ".join(meta))

                probs = {k: row.get(k, np.nan) for k in ["prob_Select","prob_Choice","prob_Prime"] if k in row}
                if probs:
                    st.progress(min(0.999, float(probs.get("prob_Select",0.0))), text=f"Select {probs.get('prob_Select',0.0):.2f}")
                    st.progress(min(0.999, float(probs.get("prob_Choice",0.0))), text=f"Choice {probs.get('prob_Choice',0.0):.2f}")
                    st.progress(min(0.999, float(probs.get("prob_Prime",0.0))),  text=f"Prime  {probs.get('prob_Prime',0.0):.2f}")

# ==================== WALKTHROUGH VIEW ====================
if view == "🧭 Guided Walkthrough":
    st.subheader("Image → Segmentation → Features → Grade")

    # Input
    c1, c2 = st.columns([2, 1])
    with c1:
        uploaded = st.file_uploader(
            "Upload a steak/cultivated-marbling image (PNG/JPG)",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=False
        )
    with c2:
        pick_from_table = None
        if "image" in df.columns:
            pick_from_table = st.selectbox(
                "…or pick one from the dataset",
                ["(none)"] + df["image"].astype(str).tolist(),
                index=0
            )

    # Controls
    cc1, cc2, cc3, cc4 = st.columns([1, 1, 1, 1])
    with cc1:
        alpha = st.slider(
            "Segmentation α (Lab-a weight)",
            0.00, 0.60, 0.25, 0.01,
            help="Robustness check: higher α penalises redder tissue → smaller fat mask."
        )
    with cc2:
        overlay_opacity = st.slider("Mask overlay opacity", 0.0, 1.0, 0.35, 0.05)
    with cc3:
        show_edges = st.checkbox("Show mask edges", value=False)
    with cc4:
        step = st.slider("Stage (1→3)", 1, 3, 3, help="Reveal preprocessing stages progressively.")

    # Acquire image
    rgb = None
    img_name = None
    if uploaded is not None:
        img_name = uploaded.name
        rgb = np.array(Image.open(uploaded).convert("RGB"))
    elif pick_from_table and pick_from_table != "(none)":
        row = df[df["image"] == pick_from_table].iloc[0].to_dict()
        path_guess = img_path(row, images_root)
        if path_guess and os.path.exists(path_guess):
            img_name = os.path.basename(path_guess)
            rgb = np.array(Image.open(path_guess).convert("RGB"))
        else:
            st.warning("Could not locate image on disk for the selected row.")

    if rgb is None:
        st.info("Upload an image or pick one from the dataset to start the walkthrough.")
        st.stop()

    # Pipeline
    gray_eq, a_ch, _ = mvp.preprocess_rgb_to_gray(rgb)
    mask = segment_fat_with_alpha(gray_eq, a_ch, alpha=alpha, cfg=mvp.CFG)

    # Stage visuals
    gray_u8 = np.clip((gray_eq * 255).round().astype(np.uint8), 0, 255)
    mask_u8 = (mask.astype(np.uint8) * 255)

    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown("### 1) RGB (input)")
        st.image(rgb, caption=img_name, width="stretch")
    with s2:
        st.markdown("### 2) Preprocessed (L-CLAHE)")
        if step >= 2:
            st.image(gray_u8, caption="Equalised luminance", width="stretch")
        else:
            st.empty()
    with s3:
        st.markdown("### 3) Mask fat")
        if step >= 3:
            area_pct = 100.0 * mask.mean()
            st.image(mask_u8, caption=f"Binary mask • fat area {area_pct:.1f}%", width="stretch")
        else:
            st.empty()

    # Features (must mirror training)
    props = mvp.measure.regionprops(mvp.measure.label(mask))
    feats = {}
    feats.update(mvp.features_area_components(mask, props))
    feats.update(mvp.features_glcm(gray_eq))
    feats.update(mvp.features_orientation_dispersion(mask))
    feats.update(mvp.features_lacunarity(mask))
    feats["meat_pct"] = 100.0 - feats.get("area_pct", 0.0)

    # Overlay (explain where the mask lies)
    st.markdown("#### Mask overlay")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(rgb)
    if show_edges:
        edges = seg.find_boundaries(mask, mode="outer")
        edge_rgb = np.dstack([np.zeros_like(edges), edges.astype(float), np.zeros_like(edges)])
        ax.imshow(edge_rgb, alpha=1.0)
    else:
        # green translucent overlay (float)
        overlay = np.dstack([
            np.zeros_like(mask, dtype=np.float32),  # R
            np.ones_like(mask,  dtype=np.float32),  # G
            np.zeros_like(mask, dtype=np.float32)   # B
        ])
        ax.imshow(overlay, alpha=overlay_opacity * mask.astype(np.float32))
    ax.axis("off")
    st.pyplot(fig, width="stretch")

    # Predict (if model available)
    probs_df = None
    pred_label = None
    confidence = np.nan
    bundle = bundle  # already loaded
    if bundle is not None:
        mdl   = bundle["model"]
        fcols = bundle["features"]
        labs  = bundle["labels"]
        xrow = {k: np.nan for k in fcols}
        for k, v in feats.items():
            if k in xrow: xrow[k] = v
        X = np.array([[xrow[k] for k in fcols]])
        pred_label = mdl.predict(X)[0]
        probs = mdl.predict_proba(X)[0]
        class_list = list(mdl.classes_)
        order = [class_list.index(c) for c in labs]
        probs = probs[order]
        probs_df = {f"prob_{c}": float(p) for c, p in zip(labs, probs)}
        confidence = float(np.max(probs))

    # Features → Grade
    st.markdown("#### Features → Grade")
    key_feats = {
        "area_pct": feats.get("area_pct", np.nan),
        "n_flecks": feats.get("n_flecks", np.nan),
        "orientation_anisotropy": feats.get("orientation_anisotropy", np.nan),
        "lacunarity_mean": feats.get("lacunarity_mean", np.nan),
        "glcm_contrast": feats.get("glcm_contrast", np.nan),
        "meat_pct": feats.get("meat_pct", np.nan),
    }
    feature_values = {k: float(v) if np.isfinite(v) else np.nan for k, v in key_feats.items()}
    table_html = render_feature_table_html(feature_values)
    row_count = len(feature_values) + 1  # +1 for header
    approx_row_px = 38                   # rough row height
    st_html(table_html, height=min(600, 80 + row_count * approx_row_px), scrolling=False)
    st.caption("Tip: hover over any feature name to see its definition.")

    if probs_df is not None:
        st.markdown("---")
        st.markdown(f"**Predicted grade:** `{pred_label}` • **Confidence:** `{confidence:.2f}`")
        st.progress(min(0.999, float(probs_df.get("prob_Select", 0.0))), text=f"Select {probs_df.get('prob_Select', 0.0):.2f}")
        st.progress(min(0.999, float(probs_df.get("prob_Choice", 0.0))), text=f"Choice {probs_df.get('prob_Choice', 0.0):.2f}")
        st.progress(min(0.999, float(probs_df.get("prob_Prime", 0.0))),  text=f"Prime  {probs_df.get('prob_Prime', 0.0):.2f}")
    else:
        st.info("Model bundle not loaded — features shown only.")
