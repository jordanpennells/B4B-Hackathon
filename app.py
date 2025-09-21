# app.py
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
    initial_sidebar_state="collapsed",
)

# -------------------- Feature glossary (hover tooltips) --------------------
FEATURE_DEFS = {
    "fat_within_steak_pct": "Fat area as a percentage of the steak foreground mask (recommended).",
    "meat_within_steak_pct": "Lean area within steak = 100 − fat_within_steak_pct.",
    "area_pct": "Fat area as a percentage of the entire image frame (legacy metric).",
    "meat_pct": "Lean area in the whole frame = 100 − area_pct (legacy).",
    "n_flecks": "Count of distinct fat regions (connected components). Many small flecks → fine marbling.",
    "orientation_anisotropy": "Directional alignment of fat structures (0–1). Higher → aligned strands; lower → isotropic.",
    "lacunarity_mean": "Measure of gaps/texture heterogeneity across scales. Higher → more irregular texture.",
    "glcm_contrast": "Texture contrast from GLCM. Higher → stronger local intensity differences.",
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
    # normalise expected columns (include Wagyu)
    for c in ["prob_Select", "prob_Choice", "prob_Prime", "prob_Wagyu"]:
        if c not in df.columns:
            df[c] = np.nan

    # confidence & similarities
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    df["confidence"] = df[prob_cols].max(axis=1) if prob_cols else np.nan
    if "prob_Prime" in df.columns and "prime_similarity" not in df.columns:
        df["prime_similarity"] = df["prob_Prime"]
    if "prob_Wagyu" in df.columns and "wagyu_similarity" not in df.columns:
        df["wagyu_similarity"] = df["prob_Wagyu"]

    # derived lean percents
    if "fat_within_steak_pct" in df.columns and "meat_within_steak_pct" not in df.columns:
        df["meat_within_steak_pct"] = 100.0 - df["fat_within_steak_pct"]
    if "meat_pct" not in df.columns and "area_pct" in df.columns:
        df["meat_pct"] = 100 - df["area_pct"]

    # convenience: image basename + lowercase join key
    if "image" not in df.columns and "path" in df.columns:
        df["image"] = df["path"].apply(os.path.basename)
    base_src = df["path"] if "path" in df.columns else df.get("image", pd.Series(dtype=str))
    df["join_base"] = base_src.astype(str).apply(lambda p: os.path.basename(p).lower())
    return df

@st.cache_data
def load_mi_scores(path="reports/mi_scores.csv"):
    if not os.path.exists(path):
        return None
    try:
        mi = pd.read_csv(path)
        # create a robust basename join key
        if "path" in mi.columns:
            src = mi["path"]
        elif "image" in mi.columns:
            src = mi["image"]
        else:
            src = pd.Series([""] * len(mi))
        mi["join_base"] = src.astype(str).apply(lambda p: os.path.basename(p).lower())
        keep = ["join_base", "MI_logit", "MI_lda", "MI_handcrafted"]
        return mi[[c for c in keep if c in mi.columns]].drop_duplicates("join_base")
    except Exception:
        return None

@st.cache_data
def load_feature_importance():
    paths = [
        "reports/feature_importance_perm.csv",  # preferred if present
        "reports/feature_importance.csv"
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                if {"feature","importance"}.issubset(df.columns):
                    return df.sort_values("importance", ascending=False).reset_index(drop=True)
            except Exception:
                pass
    return None

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

def rough_steak_mask(rgb, cfg=mvp.CFG):
    """Lightweight foreground (steak) mask for walkthrough."""
    L_eq, _, _ = mvp.preprocess_rgb_to_gray(rgb)  # [0..1]
    t = mvp.threshold_otsu(L_eq)
    fg = L_eq > t  # assume steak brighter than dark bg
    lab = mvp.measure.label(fg)
    if lab.max() == 0:
        return fg
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    keep = sizes.argmax()
    fg = (lab == keep)
    fg = mvp.closing(fg, mvp.footprint_rectangle((cfg['close_size']*2, cfg['close_size']*2)))
    fg = mvp.remove_small_objects(fg, cfg['min_obj_size'] * 10)
    return fg

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

# Pick ONE MI to surface everywhere (simple & consistent)
MI_COL = "MI_logit"

bundle = load_model_bundle()

if not os.path.exists(csv_path):
    st.warning(f"CSV not found: {csv_path}")
    st.stop()

# Load predictions CSV first
df = load_csv(csv_path).copy()

# Deduplicate predictions by basename to avoid duplicate gallery cards
if "join_base" in df.columns:
    df = df.drop_duplicates(subset=["join_base"], keep="first").reset_index(drop=True)

# Attach MI (robust basename join) and dedupe the MI table too
mi_df = load_mi_scores("reports/mi_scores.csv")
if mi_df is not None:
    mi_df = mi_df.drop_duplicates(subset=["join_base"], keep="first")
    try:
        df = df.merge(mi_df, on="join_base", how="left", validate="one_to_one")
    except Exception:
        # fallback if validation complains
        df = df.merge(mi_df, on="join_base", how="left")
else:
    for c in ["MI_logit", "MI_lda", "MI_handcrafted"]:
        df[c] = np.nan

# Pick ONE MI to surface everywhere (simple & consistent)
MI_COL = "MI_logit"

bundle = load_model_bundle()

if not os.path.exists(csv_path):
    st.warning(f"CSV not found: {csv_path}")
    st.stop()

# Load predictions CSV first
df = load_csv(csv_path).copy()

# Deduplicate predictions by basename to avoid duplicate cards
if "join_base" in df.columns:
    df = df.drop_duplicates(subset=["join_base"], keep="first").reset_index(drop=True)

# Attach MI (robust basename join) and dedupe the MI table too
mi_df = load_mi_scores("reports/mi_scores.csv")
if mi_df is not None:
    mi_df = mi_df.drop_duplicates(subset=["join_base"], keep="first")
    # one-to-one merge (will raise if duplicates slip through)
    try:
        df = df.merge(mi_df, on="join_base", how="left", validate="one_to_one")
    except Exception:
        # fall back silently if validate complains in some environments
        df = df.merge(mi_df, on="join_base", how="left")
else:
    for c in ["MI_logit", "MI_lda", "MI_handcrafted"]:
        df[c] = np.nan



# -------------------- Title & Tabs --------------------
st.title("AI Marbling Grader – Dashboard")

tab_walk, tab_gallery, tab_features, tab_summary = st.tabs([
    "🧭 Guided Walkthrough", "🖼️ Gallery", "📊 Feature Viz", "🧾 Summary"
])

# ==================== GALLERY ====================
with tab_gallery:
    fat_col = "fat_within_steak_pct" if "fat_within_steak_pct" in df.columns else "area_pct"

    cols = st.columns(5)
    with cols[0]:
        st.metric("Rows", len(df))
    with cols[1]:
        if "predicted" in df.columns:
            st.metric("Pred: Prime %", f"{(df['predicted']=='Prime').mean()*100:.1f}%")
    with cols[2]:
        st.metric("Pred: Wagyu %", f"{(df['predicted']=='Wagyu').mean()*100:.1f}%" if "predicted" in df.columns else "—")
    with cols[3]:
        if fat_col in df.columns:
            label = "Median fat% (in-steak)" if fat_col == "fat_within_steak_pct" else "Median fat% (frame)"
            st.metric(label, f"{df[fat_col].median():.1f}")
    with cols[4]:
        st.metric("High-confidence (≥0.8)", f"{(df['confidence']>=0.8).mean()*100:.1f}%")

    # Filters
    with st.expander("Filters", expanded=False):
        pred_opts = sorted(df["predicted"].dropna().unique().tolist()) if "predicted" in df.columns else []
        true_opts = sorted(df["label"].dropna().unique().tolist()) if "label" in df.columns else []

        sort_choices = ["confidence"]
        if MI_COL in df.columns: sort_choices.insert(0, MI_COL)
        for c in ["fat_within_steak_pct", "area_pct", "meat_within_steak_pct", "meat_pct"]:
            if c in df.columns: sort_choices.append(c)

        c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.0, 1.0])
        with c1:
            pred_sel = st.multiselect("Predicted", pred_opts, default=pred_opts)
        with c2:
            true_sel = st.multiselect("Truth (if present)", true_opts, default=true_opts)
        with c3:
            sort_by = st.selectbox("Sort by", sort_choices, index=0)
        with c4:
            ascending = st.checkbox("Ascending", value=False)

        conf_rng = st.slider("Confidence range", 0.0, 1.0, (0.0, 1.0), 0.01)

        c5, c6 = st.columns(2)
        with c5:
            page_size = st.selectbox("Items per page", [6, 9, 12, 15, 20], index=1)
        with c6:
            page = st.number_input("Page", min_value=1, value=1, step=1)

    # Apply Filter + sort
    work = df.copy()
    if "predicted" in work.columns and 'pred_sel' in locals() and pred_sel:
        work = work[work["predicted"].isin(pred_sel)]
    if "label" in work.columns and 'true_sel' in locals() and true_sel:
        work = work[work["label"].isin(true_sel)]
    work = work[(work["confidence"] >= conf_rng[0]) & (work["confidence"] <= conf_rng[1])]
    if 'sort_by' in locals() and sort_by in work.columns:
        work = work.sort_values(sort_by, ascending=ascending)

    # Cards grid
    start = (page - 1) * page_size
    end = start + page_size
    subset = work.iloc[start:end]

    st.write(f"Showing **{len(subset)}** of {len(work)} filtered rows (page {page})")

    n_cols = 3
    rows_count = math.ceil(max(1, len(subset)) / n_cols)
    for r in range(rows_count):
        cols = st.columns(n_cols)
        for c in range(n_cols):
            idx = r * n_cols + c
            if idx >= len(subset):
                continue
            row = subset.iloc[idx].to_dict()
            imgfile = img_path(row, images_root)
            with cols[c]:
                st.markdown(f"### {row.get('image','(image)')}")
                if imgfile and os.path.exists(imgfile):
                    im = safe_img(imgfile, max_w=480)
                    if im is not None:
                        st.image(im, width="stretch")

                # one clean meta line
                meta = []
                if "predicted" in row: meta.append(f"**Pred:** {row['predicted']}")
                if "label" in row:     meta.append(f"**Truth:** {row['label']}")
                meta.append(f"{badge_colour(row.get('confidence'))} **Conf:** {row.get('confidence', np.nan):.2f}")

                # prefer steak-aware fat/lean
                if "fat_within_steak_pct" in row and pd.notna(row["fat_within_steak_pct"]):
                    meta.append(f"**Meat %:** {100.0 - float(row['fat_within_steak_pct']):.1f}")
                elif "meat_pct" in row:
                    meta.append(f"**Meat %:** {float(row['meat_pct']):.1f}")

                if "prime_similarity" in row:
                    meta.append(f"**P(Prime):** {row['prime_similarity']:.2f}")
                if "wagyu_similarity" in row and not pd.isna(row.get("wagyu_similarity")):
                    meta.append(f"**P(Wagyu):** {row['wagyu_similarity']:.2f}")

                if MI_COL in row and pd.notna(row[MI_COL]):
                    meta.append(f"**MI:** {row[MI_COL]:.2f}")

                st.markdown(" • ".join(meta))

                # Probability bars
                prob_cols = [(k, row.get(k, np.nan)) for k in row.keys() if k.startswith("prob_")]
                friendly = {"prob_Select":"Select","prob_Choice":"Choice","prob_Prime":"Prime","prob_Wagyu":"Wagyu"}
                def sort_key(kv):
                    k,_ = kv
                    order = ["prob_Select","prob_Choice","prob_Prime","prob_Wagyu"]
                    return order.index(k) if k in order else 999
                for k, v in sorted(prob_cols, key=sort_key):
                    try:
                        label = friendly.get(k, k.replace("prob_",""))
                        st.progress(min(0.999, float(v if pd.notna(v) else 0.0)),
                                    text=f"{label} {0.0 if pd.isna(v) else float(v):.2f}")
                    except Exception:
                        pass

# ==================== FEATURE VIZ ====================
with tab_features:
    st.subheader("Feature visualisation")

    # pick important features (from reports/feature_importance*.csv if available)
    imp = load_feature_importance()
    if imp is not None:
        top_feats = imp["feature"].head(8).tolist()
    else:
        # sane defaults if no importance file exists
        top_feats = [c for c in [
            "fat_within_steak_pct", "n_flecks", "orientation_anisotropy",
            "lacunarity_mean", "glcm_contrast", "area_pct", "fleck_solidity",
            "lacunarity_slope"
        ] if c in df.columns][:8]

    st.caption("Mini histograms for the most important features" + ("" if imp is not None else " (default list)"))
    n = len(top_feats)
    if n == 0:
        st.info("No feature columns available to plot.")
    else:
        n_cols = 4
        n_rows = math.ceil(n / n_cols)
        for r in range(n_rows):
            cols = st.columns(n_cols)
            for i in range(n_cols):
                j = r*n_cols + i
                if j >= n: break
                feat = top_feats[j]
                with cols[i]:
                    series = pd.to_numeric(df[feat], errors="coerce").dropna()
                    fig, ax = plt.subplots(figsize=(3.2, 2.2))
                    ax.hist(series, bins=20)
                    ax.set_title(feat, fontsize=11)
                    ax.tick_params(axis='both', labelsize=8)
                    plt.tight_layout()
                    st.pyplot(fig, width="content")
                    plt.close(fig)

# ==================== SUMMARY ====================
with tab_summary:
    st.subheader("Marbling Index (MI) vs Grade")
    if MI_COL in df.columns and "label" in df.columns:
        # robust violin: only include labels with non-empty data
        present_labels = []
        data = []
        for lbl in ["Select", "Choice", "Prime", "Wagyu"]:
            s = pd.to_numeric(df.loc[df["label"] == lbl, MI_COL], errors="coerce").dropna()
            if len(s) > 0:
                present_labels.append(lbl)
                data.append(s.values)
        if len(data) == 0:
            st.info(f"No finite values available for {MI_COL}. "
                    "Did you run `python learn_mi.py --features features.csv`?")
        else:
            fig, ax = plt.subplots(figsize=(6, 3.8))
            parts = ax.violinplot(data, showmedians=True)
            ax.set_xticks(range(1, len(present_labels)+1))
            ax.set_xticklabels(present_labels)
            ax.set_ylabel(MI_COL)
            ax.set_title(f"{MI_COL} by label")
            plt.tight_layout()
            st.pyplot(fig, width="content")
            plt.close(fig)
    else:
        st.info("MI scores or labels not available for violin plot (need reports/mi_scores.csv and a 'label' column).")

    st.markdown("---")
    st.subheader("Confusion Matrix (compact)")
    if "label" in df.columns and "predicted" in df.columns:
        fixed_order = ["Select", "Choice", "Prime", "Wagyu"]
        labels = [lbl for lbl in fixed_order if ((df["label"]==lbl).any() or (df["predicted"]==lbl).any())]
        if labels:
            cm = confusion_matrix(df["label"], df["predicted"], labels=labels)
            fig, ax = plt.subplots(figsize=(4.5, 4.5))
            ax.imshow(cm, interpolation="nearest")
            ax.set_title("Confusion")
            ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels); ax.set_yticklabels(labels)
            for i in range(len(labels)):
                for j in range(len(labels)):
                    ax.text(j, i, int(cm[i, j]), ha="center", va="center")
            plt.tight_layout()
            st.pyplot(fig, width="content")
            plt.close(fig)
        else:
            st.info("No known labels present to render a confusion matrix.")
    else:
        st.info("Missing 'label' or 'predicted' columns for confusion matrix.")

# ==================== WALKTHROUGH ====================
with tab_walk:
    st.subheader("Image → Segmentation → Steak Mask → Features → Grade")

    # Sidebar pointers (placed here to avoid duplication)
    st.sidebar.caption("Open **Gallery**, **Feature Viz**, or **Summary** for other views.")
    st.sidebar.text_input("MI source (read-only)", MI_COL, disabled=True, help="This app surfaces MI_logit only to keep things simple.")
    st.sidebar.caption("If you regenerate MI, run learn_mi.py again to refresh reports/mi_scores.csv.")

    # Settings in sidebar (kept here to store images_root)
    st.sidebar.write("---")
    st.sidebar.caption("These settings apply globally.")
    # store images_root so Gallery can reuse
    st.session_state["images_root"] = st.sidebar.text_input("Images root (for walkthrough too)", value=images_root)

    # Inputs
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
    cc1, cc2, cc3, cc4, cc5 = st.columns([1, 1, 1, 1, 1])
    with cc1:
        alpha = st.slider(
            "Segmentation α (Lab-a weight)",
            0.00, 0.60, 0.25, 0.01,
            help="Higher α penalises redder tissue → smaller fat mask."
        )
    with cc2:
        overlay_opacity = st.slider("Mask overlay opacity", 0.0, 1.0, 0.35, 0.05)
    with cc3:
        show_edges = st.checkbox("Show mask edges", value=False)
    with cc4:
        step = st.slider("Stage (1→4)", 1, 4, 4, help="Reveal preprocessing stages progressively.")
    with cc5:
        show_insteak_only = st.checkbox("Show in-steak fat% only", value=True)

    # Acquire image
    rgb = None
    img_name = None
    if uploaded is not None:
        img_name = uploaded.name
        rgb = np.array(Image.open(uploaded).convert("RGB"))
    elif pick_from_table and pick_from_table != "(none)":
        row = df[df["image"] == pick_from_table].iloc[0].to_dict()
        path_guess = img_path(row, st.session_state["images_root"])
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
    fat_mask = segment_fat_with_alpha(gray_eq, a_ch, alpha=alpha, cfg=mvp.CFG)
    steak_mask = rough_steak_mask(rgb, cfg=mvp.CFG)

    # Stage visuals
    gray_u8 = np.clip((gray_eq * 255).round().astype(np.uint8), 0, 255)
    fat_u8 = (fat_mask.astype(np.uint8) * 255)
    steak_u8 = (steak_mask.astype(np.uint8) * 255)

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown("### 1) RGB (input)")
        st.image(rgb, caption=img_name, width="stretch")
    with s2:
        st.markdown("### 2) Preprocessed")
        if step >= 2:
            st.image(gray_u8, caption="Equalised luminance", width="stretch")
        else:
            st.empty()
    with s3:
        st.markdown("### 3) Steak mask")
        if step >= 3:
            st.image(steak_u8, caption="Steak foreground (binary)", width="stretch")
        else:
            st.empty()
    with s4:
        st.markdown("### 4) Fat mask")
        if step >= 4:
            if steak_mask.sum() > 0:
                fat_insteak_pct = 100.0 * (fat_mask & steak_mask).sum() / steak_mask.sum()
            else:
                fat_insteak_pct = np.nan
            fat_caption = (
                f"Fat (binary) • in-steak fat {fat_insteak_pct:.1f}%"
                if show_insteak_only else f"Fat (binary) • frame fat {100.0*fat_mask.mean():.1f}%"
            )
            st.image(fat_u8, caption=fat_caption, width="stretch")
        else:
            st.empty()

    # Features (display only)
    # Features (display only) — mirror training: use fat ∩ steak for morphology
    fat_in_steak = fat_mask & steak_mask

    # Area & region stats on fat within steak
    props = mvp.measure.regionprops(mvp.measure.label(fat_in_steak))
    feats = {}
    feats.update(mvp.features_area_components(fat_in_steak, props))

    # Texture on steak region (not on fat mask)
    feats.update(mvp.features_glcm_inside_steak(gray_eq, steak_mask))

    # Orientation & lacunarity on fat within steak
    feats.update(mvp.features_orientation_dispersion(fat_in_steak))
    feats.update(mvp.features_lacunarity(fat_in_steak, mvp.CFG['lacunarity_scales']))

    # Steak-aware percentages
    if steak_mask.sum() > 0:
        fat_insteak_pct = 100.0 * fat_in_steak.sum() / steak_mask.sum()
        feats["fat_within_steak_pct"] = float(fat_insteak_pct)
        feats["meat_within_steak_pct"] = float(100.0 - fat_insteak_pct)
    else:
        feats["fat_within_steak_pct"] = np.nan
        feats["meat_within_steak_pct"] = np.nan

    # Legacy “frame” fat% for reference (model doesn’t use this)
    feats["area_pct"] = float(100.0 * fat_mask.mean())


    # Overlay (explain where the mask lies)
    st.markdown("#### Mask overlay")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(rgb)
    if show_edges:
        edges = seg.find_boundaries(fat_mask, mode="outer")
        edge_rgb = np.dstack([np.zeros_like(edges, dtype=np.float32),
                              edges.astype(np.float32),
                              np.zeros_like(edges, dtype=np.float32)])
        ax.imshow(edge_rgb, alpha=1.0)
    else:
        overlay = np.dstack([
            np.zeros_like(fat_mask, dtype=np.float32),
            np.ones_like(fat_mask,  dtype=np.float32),
            np.zeros_like(fat_mask, dtype=np.float32)
        ])
        ax.imshow(overlay, alpha=overlay_opacity * fat_mask.astype(np.float32))
    ax.axis("off")
    st.pyplot(fig, width="content")
    plt.close(fig)

    # Predict (if model available)
    probs_df = None
    pred_label = None
    confidence = np.nan
    if bundle is not None:
        mdl   = bundle["model"]
        fcols = bundle["features"]
        labs  = bundle["labels"]  # e.g. ["Select","Choice","Prime","Wagyu"]
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
        "fat_within_steak_pct": feats.get("fat_within_steak_pct", np.nan),
        "meat_within_steak_pct": feats.get("meat_within_steak_pct", np.nan),
        "area_pct": feats.get("area_pct", np.nan),
        "n_flecks": feats.get("n_flecks", np.nan),
        "orientation_anisotropy": feats.get("orientation_anisotropy", np.nan),
        "lacunarity_mean": feats.get("lacunarity_mean", np.nan),
        "glcm_contrast": feats.get("glcm_contrast", np.nan),
        "meat_pct": 100.0 - feats.get("area_pct", 0.0) if np.isfinite(feats.get("area_pct", np.nan)) else np.nan,
    }
    feature_values = {k: float(v) if np.isfinite(v) else np.nan for k, v in key_feats.items()}
    table_html = render_feature_table_html(feature_values)
    row_count = len(feature_values) + 1
    approx_row_px = 38
    st_html(table_html, height=min(600, 80 + row_count * approx_row_px), scrolling=False)
    st.caption("Tip: hover over any feature name to see its definition.")

    if probs_df is not None:
        st.markdown("---")
        st.markdown(f"**Predicted grade:** `{pred_label}` • **Confidence:** `{confidence:.2f}`")
        friendly = {"prob_Select":"Select","prob_Choice":"Choice","prob_Prime":"Prime","prob_Wagyu":"Wagyu"}
        order = ["prob_Select","prob_Choice","prob_Prime","prob_Wagyu"]
        for key in order:
            if key in probs_df:
                p = probs_df[key]
                st.progress(min(0.999, float(p)), text=f"{friendly[key]} {p:.2f}")
    else:
        st.info("Model bundle not loaded — features shown only.")

# -------------------- Sidebar repeats (bottom) --------------------
st.sidebar.caption("Use **Guided Walkthrough** to test single images, **Gallery** for bulk QA, **Feature Viz** for histograms, and **Summary** for MI + confusion.")
