# 🌐 Data Augmentation & Transfer Learning for Indian Language Models

## 📖 Project Overview

This project presents a multilingual Neural Machine Translation (NMT) system designed to improve translation quality for low-resource Indian languages using **data augmentation** and **transfer learning** techniques.

The system supports bidirectional translation for:

- English ↔ Telugu  
- English ↔ Kannada  

The project evaluates multiple transformer-based models including:

- mBART50  
- NLLB (No Language Left Behind)  

A **Streamlit-based web application** is developed to provide real-time translation, evaluation, and performance visualization.

---

## 🚀 Features

- Multilingual translation (EN ↔ TE, EN ↔ KN)
- Data augmentation techniques:
  - Back-Translation
  - EDA (Easy Data Augmentation)
  - Contextual Paraphrasing
- Transfer learning using pretrained models
- Comparison of 4 model variants
- Real-time translation via web interface
- CSV batch translation support
- Evaluation metrics:
  - BLEU (SacreBLEU)
  - chrF++
  - METEOR
  - Inference Time
- Graphical performance visualization dashboard

---

## 🏗️ Project Structure
Data-Augmentation-Transfer-Learning-for-Indian-Language-Models/
│── app.py
│── train_nllb.py
│── retrain_all_models.py
│── prepare_training_files.py
│── local_eval.py
│── requirements.txt
│── README.md
│── .gitignore
│
├── data/
│ ├── final/
│ ├── eval/
│
├── outputs/
│ ├── bleu.png
│ ├── chrf.png
│ ├── meteor.png


---

## ⚙️ Installation

### 1️⃣ Clone the Repository
git clone https://github.com/sakethkumardevatha/Data-Augmentation-Transfer-Learning-for-Indian-Language-Models.git
cd Data-Augmentation-Transfer-Learning-for-Indian-Language-Models

---

### 2️⃣ Create Virtual Environment (Recommended)
python -m venv venv
Activate environment:
**Windows:**
venv\Scripts\activate
**Linux/Mac:**
source venv/bin/activate

---

### 3️⃣ Install Dependencies
pip install -r requirements.txt

---

### 4️⃣ Setup NLTK Resources (Required for METEOR)
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

---

## 📂 Dataset Setup

Ensure dataset is structured as:
data/
├── final/
├── eval/

- `final/` → cleaned training dataset  
- `eval/` → test dataset  

---

## 🤖 Model Setup

### Option 1 — Use Pretrained Models (Recommended)

Models (mBART50 / NLLB) are automatically downloaded from Hugging Face.

---

### Option 2 — Train Models
python train_nllb.py
or
python retrain_all_models.py

---

## 📊 Evaluation

Run evaluation script:
python local_eval.py

Metrics generated:

- BLEU Score  
- chrF Score  
- METEOR Score  

---

## 🖥️ Run Web Application
streamlit run app.py

Open in browser:
http://localhost:8501


---

## 🧠 Usage

The web interface allows users to:

- Select model (mBART50 / NLLB)
- Choose language direction:
  - EN → TE
  - TE → EN
  - EN → KN
  - KN → EN
- Enter custom text for translation
- Upload CSV files for batch translation
- View evaluation metrics:
  - BLEU
  - chrF
  - METEOR
  - Inference time
- Visualize model comparison graphs

---

## 📈 Output

The system generates:

- Translated text  
- Evaluation metrics  
- Graphical performance charts  

---

## 🧪 Technologies Used

- Python  
- PyTorch  
- Hugging Face Transformers  
- Hugging Face Datasets  
- Streamlit  
- Pandas  
- SacreBLEU  
- NLTK (for METEOR)  
- Matplotlib / Plotly  

---

## ⚡ GPU Support

- Recommended: CUDA-enabled GPU  
- Training performed using **Google Colab**  
- Supports batch inference optimization  

---

## ⚠️ Notes

- First run will download models (internet required)  
- GPU memory may limit batch size  
- Clear cache if CUDA out-of-memory occurs  

---

## 📌 Future Work

- Extend to more Indian languages  
- Implement LoRA-based fine-tuning  
- Deploy cloud API  
- Add speech-to-text translation  
- Domain-specific translation  

---


## 📜 License

This project is developed for academic purposes.
