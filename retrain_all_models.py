import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)

BASE_NLLB = "facebook/nllb-200-distilled-600M"

data_path = "data/kn_aug.csv"
out_dir = "models/nllb_kn_finetuned"

print("Training:", out_dir)

df = pd.read_csv(data_path).dropna()
dataset = Dataset.from_pandas(df)

tokenizer = AutoTokenizer.from_pretrained(BASE_NLLB)

# NLLB language setup
tokenizer.src_lang = "eng_Latn"
tokenizer.tgt_lang = "kan_Knda"

def preprocess(x):
    return tokenizer(
        x["en"],
        text_target=x["kn"],
        truncation=True,
        max_length=128
    )

dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

model = AutoModelForSeq2SeqLM.from_pretrained(BASE_NLLB)

# 🔥 MEMORY SAFE SETTINGS
args = Seq2SeqTrainingArguments(
    output_dir=out_dir,
    per_device_train_batch_size=1,      # VERY IMPORTANT
    gradient_accumulation_steps=16,     # keeps effective batch same
    num_train_epochs=3,
    fp16=True,
    save_strategy="no",                 # DISABLE checkpoint saving
    logging_steps=20
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model)
)

trainer.train()

model.save_pretrained(out_dir)
tokenizer.save_pretrained(out_dir)

print("NLLB Kannada training complete.")

model.to("cpu")
del model
del trainer
torch.cuda.empty_cache()
