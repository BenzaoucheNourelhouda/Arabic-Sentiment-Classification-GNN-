# Arabic Sentiment Classification with Graph Neural Networks

Classifying Arabic company reviews as **negative**, **neutral** or **positive** with a Graph Convolutional Network (GCN) built on a TextGCN-style document–word graph.

## 1. Introduction

Most text classifiers read a review as a flat sequence of words. Arabic makes that hard: rich morphology, a large vocabulary, right-to-left script and many dialects. In this project we test a different idea: instead of reading each review in isolation, we build **one graph over the whole corpus** where reviews and words are nodes, and let a GCN learn from how they connect.

We use the [Arabic Company Reviews](https://www.kaggle.com/) dataset from Kaggle (about 40,000 reviews labelled -1 / 0 / 1) and train the model to predict the sentiment of each review node. The trained model is also wrapped in a small **Streamlit** app where you can type an Arabic review and see the prediction.

**Headline result:** 81.76% accuracy on a held-out 20% test split (8,009 reviews). See [Results](#results) for the per-class breakdown, including the weak neutral class.

> University project — Saad Dahlab Blida University, Faculty of Sciences, Computer Science Department (NLP course, 2025).

## 2. Technologies

| Purpose | Tools |
|---|---|
| Language | Python 3.11 |
| Graph learning | PyTorch, PyTorch Geometric (`GCNConv`, `BatchNorm`) |
| Word embeddings | FastText Arabic vectors (`cc.ar.300`) loaded with gensim |
| Data and evaluation | pandas, NumPy, scikit-learn |
| Demo app | Streamlit, pickle |
| Environment | Kaggle Notebooks (GPU/CPU) |

## 3. Features

- **Corpus-level graph**: every review and every unique word is a node; a review is linked to the words it contains (edges in both directions).
- **Pretrained Arabic embeddings**: word nodes use 300-d FastText vectors; review nodes use the average of their words' vectors.
- **3-layer GCN** (300 → 128 → 64 → 3) with BatchNorm, ReLU and dropout (0.5).
- **Arabic text normalization**: removes non-Arabic characters, diacritics and tatweel, and unifies letter variants (أ/إ/آ → ا, ى → ي, ة → ه, and so on).
- **Three-class output**: negative / neutral / positive.
- **Reusable artifacts**: model weights, vocabulary and embedding matrix are saved so predictions can run without retraining.
- **Streamlit demo**: paste an Arabic review and get the predicted sentiment instantly.

## 4. Process

1. **Load and label** — read `CompanyReviews.csv`, drop empty reviews, map ratings `-1 / 0 / 1` to classes `0 / 1 / 2`.
2. **Clean** — apply the Arabic normalization function above.
3. **Tokenize and build the vocabulary** — whitespace tokenization; 42,930 unique words.
4. **Embeddings** — load `cc.ar.300` FastText vectors; words missing from FastText get a random vector.
5. **Build the graph** — nodes are reviews followed by words; edges connect each review to its words, in both directions. Review features are the mean of their word vectors.
6. **Split** — stratified 80/20 train/test split over review nodes (`random_state=42`).
7. **Train** — the whole graph is passed at once (transductive setting); the loss is computed only on training review nodes.

   | Setting | Value |
   |---|---|
   | Optimizer | Adam, lr = 0.01 |
   | Loss | CrossEntropyLoss |
   | Epochs | 200 (evaluated every 10) |
   | Dropout | 0.5 |
   | Seed | 42 |

8. **Evaluate and save** — keep the best checkpoint, print a classification report, and save `gnn_model.pt`, `vocab.pkl` and `embedding_matrix.npy`.
9. **Serve** — the Streamlit app loads those artifacts, cleans the input review, builds a small graph for it and returns the predicted class.

### Results

Best test accuracy: **81.76%** (epoch 190).

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Negative (سلبي) | 0.77 | 0.80 | 0.78 | 2,840 |
| Neutral (محايد) | 0.20 | 0.01 | 0.01 | 385 |
| Positive (إيجابي) | 0.85 | 0.89 | 0.87 | 4,784 |
| **Overall** | | | | accuracy 0.82, macro F1 0.56 |

The model is strong on positive and negative reviews but almost never predicts neutral. The dataset is heavily imbalanced (neutral is under 5% of the test set), so accuracy alone looks better than the model really is.

### Running it

```bash
pip install torch torch-geometric gensim scikit-learn pandas numpy streamlit
```

1. Run the notebook to train the model and generate `gnn_model.pt`, `vocab.pkl` and `embedding_matrix.npy`. The notebook downloads the FastText vectors and expects the dataset at `/kaggle/input/companyreviewsds/CompanyReviews.csv`; change the path if you run it elsewhere.
2. Launch the demo:

```bash
streamlit run app.py
```

## 5. What we learned

- We tried an approach that is still rare for Arabic: treating text classification as a **graph problem** with a GNN, instead of the usual sequence models.
- Feature engineering matters a lot in this setup: how the text is normalized, how nodes are represented (FastText vectors, averaged document features) and how the graph is built all shape what the GCN can learn.
- Accuracy can hide problems. The 82% looks good, but the per-class report shows the neutral class is essentially missed.

## 6. What could be improved

- **Neutral class**: use class weights, oversampling or a weighted loss, and report macro F1 alongside accuracy.
- **Proper validation**: the best checkpoint is currently chosen on the test set, so the reported accuracy is slightly optimistic. Add a separate validation split.
- **Richer graph**: add TF-IDF edge weights and word–word edges (PMI over a sliding window), as in the original TextGCN.
- **Better Arabic preprocessing**: stop-word removal, an Arabic-aware tokenizer or lemmatizer, and dialect handling.
- **Baselines**: compare against Logistic Regression, an LSTM and a pretrained model such as AraBERT to measure the real gain from the graph.
- **Consistency**: use the same tokenizer at training and inference time, and load the trimmed FastText file that is already prepared.
- **Inductive inference**: the GCN is trained transductively; an inductive model (for example GraphSAGE) would handle new reviews more naturally.
- **Packaging**: move the notebook code into scripts and add a `requirements.txt`.
