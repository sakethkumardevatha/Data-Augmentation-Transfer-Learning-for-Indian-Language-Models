# ======================================================
# MEMORY FIX (VERY IMPORTANT — MUST BE FIRST)
# ======================================================
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ======================================================
# IMPORTS
# ======================================================
import streamlit as st
import torch
import pandas as pd
import sacrebleu
import unicodedata
import matplotlib.pyplot as plt
import time

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
from nltk.translate.meteor_score import meteor_score

# ======================================================
# CONFIG
# ======================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 128
NUM_BEAMS = 4
BATCH_SIZE = 4

MODEL_PATHS = {
    "Telugu": {
        "Baseline": "facebook/mbart-large-50-many-to-many-mmt",
        "MBART No Aug": "models/mbart_no_aug",
        "MBART Aug": "models/mbart_te_finetuned",
        "NLLB": "models/nllb_te_finetuned"
    },
    "Kannada": {
        "Baseline": "facebook/mbart-large-50-many-to-many-mmt",
        "MBART No Aug": "models/mbart_kn_no_aug",
        "MBART Aug": "models/mbart_kn_finetuned",
        "NLLB": "models/nllb_kn_finetuned"
    }
}

def normalize(text):
    return unicodedata.normalize("NFKC", text).strip().lower()

# ======================================================
# MODEL LOADER
# ======================================================
@st.cache_resource
def load_model_and_tokenizer(path, name, target_col):

    config = AutoConfig.from_pretrained(path)
    if getattr(config, "early_stopping", None) is None:
        config.early_stopping = True

    if "nllb" in name.lower():
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
        tokenizer.src_lang = "eng_Latn"
        tgt_lang = "tel_Telu" if target_col == "te" else "kan_Knda"
        forced_bos = tokenizer.lang_code_to_id[tgt_lang]

        model = AutoModelForSeq2SeqLM.from_pretrained(
            path,
            config=config,
            torch_dtype=torch.float32
        ).to(DEVICE)

    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            path,
            config=config,
            torch_dtype=torch.float16
        ).to(DEVICE)

        if name == "Baseline":
            tokenizer = AutoTokenizer.from_pretrained(
                "facebook/mbart-large-50-many-to-many-mmt"
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)

        tokenizer.src_lang = "en_XX"
        tokenizer.tgt_lang = "te_IN" if target_col == "te" else "kn_IN"
        forced_bos = None

    return model, tokenizer, forced_bos

# ======================================================
# PAGE SETUP
# ======================================================
st.set_page_config(page_title="Indic MT Evaluation", layout="wide")
st.title("📘 Indic Project – Multilingual MT Evaluation Dashboard")

# ======================================================
# UPLOAD + CACHE CLEAR
# ======================================================
uploaded_file = st.file_uploader(
    "Upload CSV (columns: en, te OR en, kn)",
    type=["csv"]
)

clear_col1, clear_col2 = st.columns([1,6])

with clear_col1:
    if st.button("🧹"):
        st.cache_resource.clear()
        st.cache_data.clear()
        torch.cuda.empty_cache()
        st.session_state.clear()

with clear_col2:
    if st.session_state.get("cache_cleared", False):
        st.markdown("✔ Cache Cleared")

# ======================================================
# DATA LOAD
# ======================================================
if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.download_button(
        "⬇ Download Uploaded Dataset",
        df.to_csv(index=False).encode("utf-8"),
        "uploaded_dataset.csv",
        "text/csv"
    )

    if "te" in df.columns:
        lang_name = "Telugu"
        target_col = "te"
    elif "kn" in df.columns:
        lang_name = "Kannada"
        target_col = "kn"
    else:
        st.error("Dataset must contain 'te' or 'kn'")
        st.stop()

    st.success(f"Detected language: {lang_name}")
    st.info(f"Dataset size: {len(df)} sentence pairs")

    if st.button("▶ Run Evaluation"):

        df = df.dropna(subset=["en", target_col])

        sources = df["en"].astype(str).tolist()
        references_raw = df[target_col].astype(str).tolist()
        references = [[normalize(r)] for r in references_raw]

        results = {}
        predictions_store = {}
        times = {}

        total_models = len(MODEL_PATHS[lang_name])
        overall_progress = st.progress(0)

        for m_idx, (name, path) in enumerate(MODEL_PATHS[lang_name].items()):

            torch.cuda.empty_cache()

            st.write(f"Running model: {name}")
            start_time = time.time()

            model_progress = st.progress(0)

            model, tokenizer, forced_bos = load_model_and_tokenizer(path, name, target_col)

            predictions = []
            total_batches = len(range(0, len(sources), BATCH_SIZE))

            for b_idx, i in enumerate(range(0, len(sources), BATCH_SIZE)):
                batch = sources[i:i+BATCH_SIZE]

                inputs = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=MAX_LEN
                )

                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

                with torch.no_grad():

                    if "nllb" in name.lower():
                        bos = forced_bos
                    elif "mbart" in name.lower():
                        bos = tokenizer.lang_code_to_id.get(tokenizer.tgt_lang, None)
                    else:
                        bos = None

                    outputs = model.generate(
                        **inputs,
                        max_length=MAX_LEN,
                        num_beams=NUM_BEAMS,
                        forced_bos_token_id=bos
                    )

                predictions.extend([
                    normalize(tokenizer.decode(o, skip_special_tokens=True))
                    for o in outputs
                ])

                model_progress.progress((b_idx+1)/total_batches)

            bleu = sacrebleu.corpus_bleu(predictions, references, smooth_method="exp")
            chrf = sacrebleu.corpus_chrf(predictions, references)

            meteor = sum(
                meteor_score([references_raw[i].split()], predictions[i].split())
                for i in range(len(predictions))
            ) / len(predictions)

            results[name] = {
                "BLEU": round(bleu.score, 2),
                "chrF++": round(chrf.score, 2),
                "METEOR": round(meteor * 100, 2)
            }

            predictions_store[name] = predictions
            times[name] = round(time.time() - start_time, 2)

            model.to("cpu")
            del model
            torch.cuda.empty_cache()

            overall_progress.progress((m_idx+1)/total_models)

        # ======================================================
        # RESULTS
        # ======================================================
        st.subheader("📊 Model Comparison")
        results_df = pd.DataFrame(results).T
        st.dataframe(results_df)

        c1,c2,c3=st.columns(3)

        with c1:
            fig1,ax1=plt.subplots()
            ax1.bar(results_df.index,results_df["BLEU"])
            ax1.set_title("BLEU")
            st.pyplot(fig1)

        with c2:
            fig2,ax2=plt.subplots()
            ax2.bar(results_df.index,results_df["chrF++"])
            ax2.set_title("chrF++")
            st.pyplot(fig2)

        with c3:
            fig3,ax3=plt.subplots()
            ax3.bar(results_df.index,results_df["METEOR"])
            ax3.set_title("METEOR")
            st.pyplot(fig3)

        st.subheader("⏱ Model Inference Time (seconds)")
        st.dataframe(pd.DataFrame.from_dict(times, orient="index", columns=["Seconds"]))

        st.subheader("🔍 Sample Translations")

        sample_df = pd.DataFrame({
            "English": sources[:5],
            "Reference": references_raw[:5]
        })

        for name in predictions_store:
            sample_df[name] = predictions_store[name][:5]

        st.dataframe(sample_df)

        out_df = df.copy()
        for name in predictions_store:
            out_df[name + "_pred"] = predictions_store[name]

        st.download_button(
            "⬇ Download Predictions",
            out_df.to_csv(index=False).encode("utf-8"),
            "model_predictions.csv",
            "text/csv"
        )   