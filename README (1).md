# 📰 Fake News Detector — DistilBERT + Explainable AI

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

A fine-tuned **DistilBERT** model that classifies news articles as **Real** or **Fake** with ~99% accuracy, paired with an interactive **Streamlit** web app featuring word-level explainability via **Integrated Gradients** (Captum).

---

## 🖼️ App Preview

| Main Classifier | Word-Level Explanation |
|---|---|
| Paste any article → instant verdict + confidence gauge | Token highlights show exactly which words drove the prediction |

---

## ✨ Features

- **Binary classification** — Real (1) vs Fake (0) with softmax confidence scores
- **Explainable AI** — Integrated Gradients via Captum highlights which tokens support or contradict the verdict
- **Credibility heuristics** — Rule-based alerts for ALL-CAPS abuse, excessive exclamation marks, and known sensationalist phrases
- **Performance dashboard** — Confusion matrix, per-class precision/recall/F1, test accuracy
- **Word clouds** — Visual vocabulary comparison between real and fake training articles
- **Prediction history** — Session log of all analysed articles

---

## 🗂️ Project Structure

```
fake-news-detector/
│
├── train_fake_news.py          # Colab training script (data prep → fine-tune → export)
├── app.py                      # Streamlit web application
│
├── fake_news_bert/             # Saved model & tokenizer (after training)
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer_config.json
│   └── vocab.txt
│
├── fake_news_metrics.json      # Accuracy, classification report, confusion matrix
├── wordcloud_real.png          # Word cloud — real news training data
├── wordcloud_fake.png          # Word cloud — fake news training data
│
└── requirements.txt
```

---

## 📦 Requirements

```txt
torch>=2.0.0
transformers>=4.41.0
datasets
streamlit>=1.35.0
plotly
captum
wordcloud
nltk
pandas
numpy
scikit-learn
matplotlib
```

Install everything:

```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### 1 · Train the model (Google Colab)

1. Upload `True.csv` and `Fake.csv` to your Google Drive at `MyDrive/`.
2. Open `train_fake_news.py` in a Colab notebook (GPU runtime recommended).
3. Run all cells. The script will:
   - Clean and split the data (80 / 20 stratified)
   - Fine-tune `distilbert-base-uncased` for 3 epochs with early stopping
   - Save the model to `MyDrive/fake_news_bert/`
   - Export `fake_news_metrics.json` and both word cloud PNGs

### 2 · Run the Streamlit app locally

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/fake-news-detector.git
cd fake-news-detector

# Install dependencies
pip install -r requirements.txt

# Copy model assets from Drive into the project root
# (fake_news_bert/, fake_news_metrics.json, wordcloud_*.png)

# Launch the app
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧠 Model Details

| Property | Value |
|---|---|
| Base model | `distilbert-base-uncased` |
| Parameters | ~67 M |
| Max token length | 512 |
| Training epochs | 3 (early stopping, patience = 2) |
| Batch size | 16 (train) / 64 (eval) |
| Optimiser | AdamW, weight decay = 0.01 |
| Warmup | 10% of total steps |
| Mixed precision | fp16 (auto, GPU only) |
| Test accuracy | **~99%** |

### Why DistilBERT?

DistilBERT retains ~97% of BERT's language understanding at 40% fewer parameters and 60% faster inference — ideal for a production-ready demo without sacrificing accuracy.

---

## 📊 Dataset

| Split | Articles |
|---|---|
| Real news | 21,417 |
| Fake news | 23,481 |
| **Total** | **44,898** |
| Train | 35,918 |
| Test | 8,980 |

Dataset source: [Fake and Real News Dataset — Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

Preprocessing applied:
- URL and HTML tag removal
- Punctuation normalisation
- Lowercasing
- Whitespace normalisation
- Title + body concatenation

---

## 🔍 Explainability — Integrated Gradients

The app uses **Captum's LayerIntegratedGradients** on the word-embedding layer to assign an importance score to every token:

- 🟢 **Green highlight** → token pushes prediction toward *Real*
- 🔴 **Red highlight** → token pushes prediction toward *Fake*
- Darker shade = stronger influence

The method interpolates between a padding-token baseline and the actual input, integrating gradients of the output class logit with respect to the embeddings across 50 steps.

---

## ⚠️ Limitations

- Trained on a specific English-language dataset; may not generalise to all domains or languages.
- ~99% accuracy does not mean infallible — always cross-check suspicious articles with reputable fact-checkers ([Snopes](https://snopes.com), [FactCheck.org](https://factcheck.org), [PolitiFact](https://politifact.com)).
- Satire and opinion pieces may be misclassified.

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Model | Hugging Face Transformers, PyTorch |
| Explainability | Captum (Integrated Gradients) |
| App | Streamlit, Plotly |
| Visualisation | Matplotlib, WordCloud |
| Data | Pandas, scikit-learn, NLTK |
| Training | Google Colab (T4 GPU) |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

- [Hugging Face](https://huggingface.co) for the Transformers library and pre-trained DistilBERT weights
- [Captum](https://captum.ai) by Meta AI for the interpretability framework
- [Streamlit](https://streamlit.io) for the rapid app framework
- Dataset by [Clément Bisaillon](https://www.kaggle.com/clmentbisaillon) on Kaggle
