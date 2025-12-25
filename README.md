# Fake News Detection (NLP, Text Classification)

This project is a proof-of-concept system to classify news as **fake** or **real** using text classification methods. The goal is to show that a simple, automated model can help end users judge the information they receive online (news, social posts, messages, emails, SMS), without depending only on closed or black-box platforms.

---

## Why this project

Misinformation spreads fast on the internet and it can influence public opinion and real-world events. Large platforms try to detect fake content, but their models are not open and users cannot easily understand how decisions are made. This project focuses on building a transparent, repeatable approach and documenting the full workflow from dataset to evaluation.

---

## What the project does

- Takes news data with labels (fake or real)
- Cleans and prepares the text for machine learning
- Builds and evaluates:
  - a baseline model
  - a stronger machine learning classifier
  - a deep learning model (to check if it can improve results)
- Compares the models using standard classification metrics

---

## Dataset

The project uses the **WELFake dataset** (from Kaggle), a balanced fake news dataset collected from multiple sources and published in an IEEE venue. The original dataset contains **72,134** news articles with a **51:49** split between fake and real news.

The dataset contains columns like:
- index
- title
- text
- label (0 = fake, 1 = real)

This project mainly uses the **title** as the input feature and the **label** as the target.

---

## Approach

### 1) Data preprocessing
The dataset is cleaned to make it usable for training:
- remove duplicate titles
- remove rows with missing titles
- reset indexing and keep only the needed columns

### 2) Text cleaning (making text ML-friendly)
To convert titles into a form that a model can learn from:
- remove punctuation and special characters
- tokenize into words
- remove stop words (common words like “the”, “and”, “is”)
- apply stemming (reduce words to a root form)
- store the cleaned titles for feature extraction

### 3) Feature extraction (Bag of Words)
Cleaned titles are converted into numbers using word frequency features:
- unigrams and bigrams are extracted
- the result is a matrix where rows are titles and columns represent word counts

To keep experiments fast, two working subsets are used during training:
- a smaller set (500 samples)
- a larger set (2000 samples)

---

## Models tested

### Baseline model
A simple baseline is used first to get a reference score:
- Multinomial Naive Bayes (good for word-count features)

### Main machine learning classifier
A stronger classifier is trained and tuned:
- Support Vector Machine (SVM)
- Hyperparameters are tuned using grid search (C, gamma, kernel)

### Deep learning experiment
A deep learning model is added to check if it improves performance:
- LSTM-based text classifier
- Text is encoded using a TextVectorization layer
- Trained for multiple epochs and evaluated on test data

---

## Evaluation

Models are evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix (to understand false positives and false negatives)

---

## Key findings

- The baseline model (Naive Bayes) performed strongly and improved when dataset size increased.
- The SVM model performed better on the smaller dataset, but its accuracy slightly dropped when moving from 500 to 2000 samples.
- The LSTM deep learning model produced a similar accuracy to the machine learning models, but it did not clearly outperform them in this setup.
- Simple models can match the performance of more complex models, so it is always worth comparing against a baseline.

---

## Limitations

- Due to compute limits, the models (especially deep learning) were not trained on the full dataset size.
- Results are based on the sampled subsets used in the experiments.

---

## Future work

- Train and evaluate on the full dataset
- Improve deep learning performance with better tuning and larger training data
- Package the classifier as a backend service for a real application (web/mobile)
- Share the code as an open-source project so the model can be reviewed and improved by others

---

## Out of scope

- Web application / mobile application
- “AI as a service” deployment (this project focuses on the proof of concept)
