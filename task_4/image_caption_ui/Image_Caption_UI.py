# app.py – Image Captioning Inference UI
# Run: streamlit run app.py
# =============================================================================

import os
import zipfile
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Import from your local model module (adjust path if needed)
from model import (
    Vocabulary, ImageCaptionDataset, MyCollate, CNNtoRNN,
    generate_caption_beam_search, transform, device,
    IMAGENET_MEAN, IMAGENET_STD,
)

# =============================================================================
# CONFIGURATION – Edit these paths to match your local setup
# =============================================================================
CHECKPOINT_PATH = "Image_Caption_best_model.pth.tar"   # model checkpoint
VOCAB_PATH      = "vocab.pkl"                          # saved vocabulary

# Model architecture (must match training)
EMBED_SIZE  = 256
HIDDEN_SIZE = 512
NUM_LAYERS  = 1

# Dataset paths (for Browse Dataset and Evaluate pages)
IMAGES_DIR    = "/home/goldeno/Downloads/image_caption_dataset_archive/Images"
CAPTIONS_FILE = "flickr8k/captions.txt"

# =============================================================================
# Streamlit Page Config
# =============================================================================
st.set_page_config(
    page_title="Image Captioning",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Helper Functions
# =============================================================================
def denorm(tensor):
    """Denormalize an image tensor for display."""
    mean = np.array(IMAGENET_MEAN)
    std  = np.array(IMAGENET_STD)
    return np.clip(std * tensor.permute(1, 2, 0).numpy() + mean, 0, 1)

def run_beam(image_or_tensor, beam_size=5):
    """Generate caption using beam search."""
    return generate_caption_beam_search(image_or_tensor, model, vocab, device, beam_size=beam_size)

# =============================================================================
# Cached Model & Dataset Loaders
# =============================================================================
@st.cache_resource(show_spinner="Loading model and vocabulary…")
def load_model():
    """Load the trained model and vocabulary."""
    # Check required files exist
    for path, label in [(VOCAB_PATH, "Vocabulary"), (CHECKPOINT_PATH, "Checkpoint")]:
        if not os.path.exists(path):
            st.error(f"{label} not found at `{path}`. Update the CONFIG block in app.py.")
            st.stop()

    vocab = Vocabulary.load(VOCAB_PATH)
    model = CNNtoRNN(EMBED_SIZE, HIDDEN_SIZE, len(vocab), NUM_LAYERS).to(device)
    ckpt  = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, vocab, ckpt.get("epoch", "?")

@st.cache_resource(show_spinner="Loading dataset splits…")
def load_datasets(vocab):
    """Load validation and test datasets from local files."""
    if not os.path.exists(CAPTIONS_FILE) or not os.path.isdir(IMAGES_DIR):
        return None, None

    df = pd.read_csv(CAPTIONS_FILE)
    tv, test_df = train_test_split(df, test_size=0.1, random_state=42)
    _, val_df   = train_test_split(tv, test_size=0.1, random_state=42)

    val_ds = ImageCaptionDataset(IMAGES_DIR, val_df,  transform, vocab)
    test_ds = ImageCaptionDataset(IMAGES_DIR, test_df, transform, vocab)
    return val_ds, test_ds

# Load model and vocabulary once
model, vocab, ckpt_epoch = load_model()

# =============================================================================
# Sidebar – Navigation and controls
# =============================================================================
with st.sidebar:
    st.title("🖼️ Image Captioning")
    st.divider()
    page = st.selectbox(
        "Navigation",
        ["🔍 Single Inference", "📦 Batch Inference", "🗃️ Browse Dataset", "📊 Evaluate"],
        label_visibility="collapsed",
    )
    st.divider()
    beam_size = st.slider("Beam size", 1, 10, 5, help="Number of candidate sequences in beam search")
    st.divider()
    st.caption(f"Checkpoint: `{os.path.basename(CHECKPOINT_PATH)}`")
    st.caption(f"Epoch: {ckpt_epoch} · Device: {device.upper()}")

# =============================================================================
# Page Functions
# =============================================================================
def page_single_inference():
    """Single image upload and caption generation."""
    st.header("Single Inference")
    st.markdown("Upload any image — the model will generate a caption using beam search.")

    uploaded = st.file_uploader(
        "Choose an image", type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded:
        col_img, col_cap = st.columns(2, gap="large")
        pil_img = Image.open(uploaded).convert("RGB")

        with col_img:
            st.image(pil_img, use_container_width=True)

        with col_cap:
            if st.button("✨ Generate caption", use_container_width=True):
                with st.spinner("Running beam search…"):
                    words = run_beam(pil_img, beam_size)
                cap = " ".join(words)
                st.success("**Generated caption**")
                st.write(cap)
                st.caption(f"{len(words)} words · beam = {beam_size}")

def page_batch_inference():
    """Batch processing of a ZIP file of images."""
    st.header("Batch Inference")
    st.markdown("Upload a **ZIP file** of images. The model captions every image and returns a CSV.")

    batch_zip = st.file_uploader("Upload ZIP file", type=["zip"], label_visibility="collapsed")

    if batch_zip:
        st.info(f"📎 `{batch_zip.name}`  ({batch_zip.size / 1024:.1f} KB)")
        if st.button("▶ Run batch captioning", type="primary"):
            results = []
            with tempfile.TemporaryDirectory() as tmpdir:
                with zipfile.ZipFile(batch_zip) as z:
                    z.extractall(tmpdir)

                img_files = sorted(
                    f for f in os.listdir(tmpdir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
                )

                if not img_files:
                    st.error("No images found in the ZIP.")
                    st.stop()

                progress_bar = st.progress(0, text="Starting…")
                preview_cols = st.columns(4)
                shown = 0

                for i, fname in enumerate(img_files):
                    pil = Image.open(os.path.join(tmpdir, fname)).convert("RGB")
                    words = run_beam(pil, beam_size)
                    cap = " ".join(words)
                    results.append({"image": fname, "caption": cap})

                    if shown < 8:
                        preview_cols[shown % 4].image(pil, caption=f"{fname}\n{cap}", use_container_width=True)
                        shown += 1

                    progress_bar.progress((i + 1) / len(img_files), text=f"{i+1}/{len(img_files)} — {fname}")

                progress_bar.empty()

            df = pd.DataFrame(results)
            st.success(f"✓ Captioned {len(df)} images.")
            st.dataframe(df, use_container_width=True)
            st.download_button(
                "⬇ Download captions.csv",
                df.to_csv(index=False).encode(),
                "captions.csv",
                "text/csv",
            )

def page_browse_dataset():
    """Browse validation samples and compare model captions with ground truth."""
    st.header("Browse Dataset")

    val_ds, _ = load_datasets(vocab)
    if val_ds is None:
        st.error("Dataset files not found. Check `IMAGES_DIR` and `CAPTIONS_FILE` in app.py.")
        return

    st.markdown(f"**{len(val_ds):,}** samples in the validation split.")
    st.divider()

    idx = st.slider("Sample index", 0, len(val_ds) - 1, 0)
    img_t, cap_t = val_ds[idx]

    # Decode ground-truth caption (skip special tokens)
    ref_words = [
        vocab.itos[t.item()] for t in cap_t
        if vocab.itos[t.item()] not in {"<PAD>", "<SOS>", "<EOS>"}
    ]
    ref_caption = " ".join(ref_words)

    col_img, col_txt = st.columns(2, gap="large")
    with col_img:
        st.image(denorm(img_t), use_container_width=True)

    with col_txt:
        st.markdown("**Ground-truth caption**")
        st.info(ref_caption)

        if st.button("✨ Generate model caption", use_container_width=True):
            with st.spinner("Running beam search…"):
                words = run_beam(img_t, beam_size)
            st.markdown("**Model caption**")
            st.success(" ".join(words))

def page_evaluate():
    """Evaluate model on test set: cross-entropy loss and BLEU-4."""
    st.header("Evaluation")
    st.markdown("Runs beam search over the test split and computes **cross-entropy loss** and **BLEU-4**.")

    _, test_ds = load_datasets(vocab)
    if test_ds is None:
        st.error("Dataset files not found. Check `IMAGES_DIR` and `CAPTIONS_FILE` in app.py.")
        return

    st.markdown(f"**{len(test_ds):,}** test samples total.")
    st.divider()

    max_samples = st.slider(
        "Samples to evaluate (reduce if no GPU)",
        50, len(test_ds), min(500, len(test_ds)), step=50,
    )

    if st.button("▶ Run evaluation", type="primary"):
        try:
            from torchmetrics.functional.text import bleu_score as _bleu
        except ImportError:
            st.error("Install torchmetrics:  `pip install torchmetrics`")
            return

        criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])
        test_loader = DataLoader(
            test_ds, batch_size=1, shuffle=False,
            collate_fn=MyCollate(vocab.stoi["<PAD>"]),
        )

        total_loss, candidates, references = 0.0, [], []
        progress_bar = st.progress(0, text="Evaluating…")

        model.eval()
        with torch.no_grad():
            for i, (imgs, caps) in enumerate(test_loader):
                if i >= max_samples:
                    break

                imgs, caps = imgs.to(device), caps.to(device)
                out = model(imgs, caps[:, :-1])
                logits = out[:, 1:, :].reshape(-1, out.shape[2])
                tgts = caps[:, 1:].reshape(-1)
                total_loss += criterion(logits, tgts).item()

                # Generate caption with beam search
                pred_words = run_beam(imgs[0], beam_size)
                ref_words = [
                    vocab.itos[idx] for idx in caps[0].tolist()
                    if vocab.itos[idx] not in {"<SOS>", "<EOS>", "<PAD>"}
                ]
                candidates.append(" ".join(pred_words))
                references.append([" ".join(ref_words)])

                progress_bar.progress((i + 1) / max_samples, text=f"Sample {i+1} / {max_samples}")

        progress_bar.empty()

        avg_loss = total_loss / max_samples
        score = float(_bleu(candidates, references, n_gram=4))

        # Display metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Test Loss (cross-entropy)", f"{avg_loss:.4f}")
        with col2:
            st.metric("BLEU-4", f"{score:.4f}")

        # Show sample predictions
        st.divider()
        st.subheader("Sample predictions")

        for i in range(min(8, len(candidates))):
            img_t, _ = test_ds[i]
            col_img, col_txt = st.columns([1, 2], gap="large")
            with col_img:
                st.image(denorm(img_t), use_container_width=True)
            with col_txt:
                st.markdown("**Model**")
                st.write(candidates[i])
                st.markdown("**Reference**")
                st.code(references[i][0])
            st.divider()

        # Download results
        res_df = pd.DataFrame({
            "predicted": candidates,
            "reference": [r[0] for r in references],
        })
        st.download_button(
            "⬇ Download eval_results.csv",
            res_df.to_csv(index=False).encode(),
            "eval_results.csv",
            "text/csv",
        )

# =============================================================================
# Main – Route to selected page
# =============================================================================
if page == "🔍 Single Inference":
    page_single_inference()
elif page == "📦 Batch Inference":
    page_batch_inference()
elif page == "🗃️ Browse Dataset":
    page_browse_dataset()
elif page == "📊 Evaluate":
    page_evaluate()