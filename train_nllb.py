# ======================================================
# MEMORY FIX (MUST BE FIRST)
# ======================================================
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ======================================================
# IMPORTS
# ======================================================
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)

# ======================================================
# CONFIG
# ======================================================
MODEL_NAME = "models/nllb_kn_finetuned"   # continue training
SRC_LANG = "eng_Latn"
TGT_LANG = "kan_Knda"   # change to tel_Telu for Telugu
DATA_PATH = "data/train.csv"

# ======================================================
# LOAD DATA
# ======================================================
df = pd.read_csv(DATA_PATH).dropna()
dataset = Dataset.from_pandas(df)

# ======================================================
# TOKENIZER
# ======================================================
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
tokenizer.src_lang = SRC_LANG
tokenizer.tgt_lang = TGT_LANG

def preprocess(example):
    model_inputs = tokenizer(
        example["en"],
        max_length=128,
        truncation=True
    )

    labels = tokenizer(
        text_target=example["kn"],   # change to "te" if Telugu
        max_length=128,
        truncation=True
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

# ======================================================
# MODEL
# ======================================================
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# IMPORTANT: saves ~35-40% GPU memory
model.gradient_checkpointing_enable()

# ======================================================
# TRAINING SETTINGS (6GB GPU SAFE)
# ======================================================
training_args = Seq2SeqTrainingArguments(
    output_dir="models/nllb_kn_finetuned",
    learning_rate=3e-5,
    per_device_train_batch_size=2,        # low memory
    gradient_accumulation_steps=8,        # effective batch size maintained
    num_train_epochs=5,
    warmup_ratio=0.05,
    weight_decay=0.01,
    fp16=True,
    predict_with_generate=True,
    logging_steps=50,
    save_strategy="epoch",
    save_total_limit=2
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# ======================================================
# TRAIN
# ======================================================
torch.cuda.empty_cache()
trainer.train()

# ======================================================
# SAVE
# ======================================================
model.save_pretrained("models/nllb_kn_finetuned")
tokenizer.save_pretrained("models/nllb_kn_finetuned")

print("Training completed and model saved.")
