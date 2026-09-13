# 🎬 Movie Review Text Vectorization — TF-IDF Feature Engineering Case Study

[![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-TF--IDF-orange?logo=scikitlearn)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter)](https://jupyter.org/)
[![Status](https://img.shields.io/badge/Status-Feature%20Engineering%20Demo-yellow)]()

A hands-on exploration of **TF-IDF (Term Frequency–Inverse Document Frequency)** vectorization applied to a movie-review text corpus, covering n-gram feature extraction and feature-importance visualization with `scikit-learn`, `pandas`, and `seaborn`.

This README documents the notebook honestly — including the bugs, scale issues, and design gaps found while auditing it — rather than presenting a polished summary that hides them. Job-ready NLP work means knowing exactly where a pipeline breaks, and this is that account.

---

## 📌 What This Project Actually Is

`movie_review_case_study_nlp.ipynb` demonstrates the **text vectorization stage** of an NLP pipeline:

- Converts raw review text into numerical features using unigram + bigram TF-IDF
- Extracts and inspects the resulting feature vocabulary
- Visualizes feature importance across documents with a Seaborn heatmap

**Important scope note:** despite the "case study" name, this notebook stops at feature engineering. There is **no sentiment labeling, no classifier, and no evaluation metric** anywhere in it. If you're expecting a trained sentiment-analysis model, it isn't here yet — see [Roadmap](#-roadmap--what-id-add-next) below.

---

## 🗂️ Dataset

The corpus is a **synthetic** set of movie review sentences, not scraped or user-generated data. On inspection:

| Metric | Value |
|---|---|
| Total review records | 100,000 |
| **Unique reviews** | **200** |
| Duplication factor | ~500× |
| TF-IDF vocabulary size (uni+bigrams) | 433 features |

The 100,000 rows are built by repeating 200 template sentence-combinations 500 times each. This is fine for a scale demo, but it isn't disclosed anywhere in the original notebook — a recruiter or teammate opening it would reasonably assume 100K genuine reviews. Worth stating explicitly, which is why it's called out here.

---

## 🛠️ Tech Stack

- **Python 3.14**
- `scikit-learn` — `TfidfVectorizer`
- `pandas`, `numpy` — data handling
- `matplotlib`, `seaborn` — visualization
- Jupyter Notebook

---

## 🔍 Methodology

1. **Vectorization** — `TfidfVectorizer(ngram_range=(1,2), stop_words=None)` turns each review into a weighted unigram + bigram vector.
2. **Feature inspection** — the fitted vocabulary (433 unigrams/bigrams) is printed for review.
3. **Per-document TF-IDF scores** — every non-zero (feature, score) pair is printed per document.
4. **Visualization** — an annotated Seaborn heatmap plots the dense TF-IDF matrix across documents and features.

---

## ⚠️ Honest Audit — Issues Found

This is the part most portfolio READMEs skip. Here's what actually surfaced when I went through the notebook cell by cell:

| # | Issue | Where | Impact |
|---|---|---|---|
| 1 | **No labels, no model, no evaluation** | Whole notebook | It's a vectorization demo, not a "case study" with a conclusion — the name overpromises |
| 2 | **`stop_words=None`** | Cell 2 (vectorizer) | Common low-information words (*the, was, and, is*) stay in the vocabulary, diluting the signal the TF-IDF weighting is supposed to surface |
| 3 | **Full-corpus score printout** | Cell 4 (per-document loop) | Looping print over all 100,000 documents generated a **53.6-million-character output cell**, which is why the `.ipynb` file on disk is **~100 MB** |
| 4 | **File size blocks GitHub push** | Whole notebook file | GitHub hard-blocks any file ≥100 MB without Git LFS, and warns above 50 MB. At ~100 MB, this file is sitting right at (likely over) that wall — **it may fail to push as-is** |
| 5 | **Heatmap not computationally feasible** | Cell 5 (Seaborn heatmap) | An annotated heatmap over a 100,000 × 433 dense matrix isn't something Seaborn can render meaningfully — the cell has no captured output, consistent with it never finishing |
| 6 | **Duplicated rows vectorized anyway** | Vectorization step | With only 200 unique sentences, transforming all 100,000 rows re-computes identical TF-IDF vectors 500× over — wasted compute for zero additional information |
| 7 | **Variable naming: `x`** | Cell 2 | The feature matrix is assigned to lowercase `x`; convention (and most sklearn docs) use uppercase `X` for a 2D feature matrix, reserving lowercase for a single sample/vector |
| 8 | **Trailing empty cell** | Last cell | Leftover empty cell — harmless, but a sign the notebook wasn't cleaned before saving |

None of this is disqualifying — it's exactly the kind of thing a code reviewer would flag, and flagging it myself is more useful to you than a recruiter finding it first.

---

## ✅ Before You Push This to GitHub

Because of issue #3/#4 above, **clear the output of the per-document print cell before committing**, or GitHub may reject the push outright:

```bash
jupyter nbconvert --clear-output --inplace movie_review_case_study_nlp.ipynb
```

This alone should drop the file from ~100 MB down to a few KB, since almost the entire file size is that one output block.

---

## 🚀 Roadmap — What I'd Add Next

- [ ] Deduplicate to the 200 unique reviews before vectorizing
- [ ] Add sentiment labels (or an existing labeled corpus, e.g. IMDB) and an actual classifier — Logistic Regression or Naive Bayes as a baseline
- [ ] Train/test split + accuracy, precision/recall, F1, confusion matrix
- [ ] Apply `stop_words="english"` (or a custom list) and compare vocabulary/feature quality before vs. after
- [ ] Replace the full-corpus print loop with a sampled preview (e.g., 5–10 documents)
- [ ] Sample the heatmap to a manageable subset of documents/features instead of the full matrix
- [ ] Add `nbstripout` as a pre-commit hook to prevent output-bloated notebooks in future commits

---

## ▶️ How to Run

```bash
git clone <your-repo-url>
cd <repo-folder>
pip install scikit-learn pandas numpy matplotlib seaborn jupyter
jupyter notebook movie_review_case_study_nlp.ipynb
```

**Recommended:** run the per-document print cell on a small slice (`reviews[:10]`) rather than the full 100,000 — the original loop is what produced the oversized notebook file described above.

---

## 🎯 Skills Demonstrated

`Text Preprocessing` `TF-IDF Vectorization` `N-gram Feature Extraction` `scikit-learn` `Data Visualization` `Debugging & Code Review` `Notebook Hygiene`

---

## 👤 About Me

**Vishnusai Vydhyam** — Final-year B.Tech CSE (AI & ML) student, building an ML/AI portfolio for fresher and internship roles.

- GitHub: [@vishnusai2005](https://github.com/vishnusai2005)
- LinkedIn: [vishnusai-vydhyam](https://www.linkedin.com/in/vishnusai-vydhyam)
- Hugging Face: [v2005](https://huggingface.co/v2005)
- X: [@VishnusaiSaii](https://x.com/VishnusaiSaii)

---

## 📄 License

Not yet specified — add a `LICENSE` file (MIT is a common choice for portfolio projects) before publishing if you want to make reuse terms explicit.
