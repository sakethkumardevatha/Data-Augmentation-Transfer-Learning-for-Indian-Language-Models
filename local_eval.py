import torch
import pandas as pd
import sacrebleu
import unicodedata
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("data/en_knn_eval.csv")

if "te" in df.columns:
    lang = "Telugu"
    tgt = "te"
elif "kn" in df.columns:
    lang = "Kannada"
    tgt = "kn"
else:
    raise ValueError("Dataset must contain te or kn")

sources = df["en"].astype(str).tolist()
references_raw = df[tgt].astype(str).tolist()
references = [[normalize(r)] for r in references_raw]

# =========================
# MODEL LOOP
# =========================
for name, path in MODEL_PATHS[lang].items():

    print(f"\nRunning: {name}")

    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(path).to(DEVICE)

    # ⭐ LANGUAGE SETTINGS
    if "nllb" in name.lower():
        tokenizer.src_lang = "eng_Latn"
        tgt_lang = "tel_Telu" if tgt == "te" else "kan_Knda"
        bos = tokenizer.lang_code_to_id[tgt_lang]
    else:
        tokenizer.src_lang = "en_XX"
        tokenizer.tgt_lang = "te_IN" if tgt == "te" else "kn_IN"
        bos = tokenizer.lang_code_to_id.get(tokenizer.tgt_lang, None)

    predictions = []

    for i in range(0, len(sources), BATCH_SIZE):
        batch = sources[i:i+BATCH_SIZE]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN
        ).to(DEVICE)

        with torch.no_grad():
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

    bleu = sacrebleu.corpus_bleu(predictions, references)
    chrf = sacrebleu.corpus_chrf(predictions, references)

    print("BLEU:", round(bleu.score, 2))
    print("chrF++:", round(chrf.score, 2))
