# Multimodal Fake News Detection using NLP, Computer Vision and BERT

## Overview

This project explores fake news detection using multiple machine learning and deep learning approaches. The project was implemented in a phased manner to understand the strengths and limitations of different textual and visual representations for misinformation detection.

### Workflow

* Classical NLP baselines (TF-IDF)
* Image feature extraction using ResNet50
* Multimodal feature fusion
* BERT-based contextual embeddings
* Experimental comparison and evaluation

---

## Dataset

This project uses the **Fake and Real News Dataset**.

### Dataset Files

* `Fake.csv`
* `True.csv`

### Each Article Contains

* Title
* News Text
* Label

### Labels

* `0` → Fake News
* `1` → Real News

---

## Notebook Descriptions

### 1. sml_nlp_baselines.ipynb

A classical NLP pipeline was developed for fake news detection using TF-IDF feature extraction and traditional machine learning classifiers. The news text was converted into numerical features. TF-IDF assigned higher importance to commonly occuring words and lower to others.
News articles were represented as weighted word vectors and used to train **Logistic Regression**, **Naive Bayes**, and **Support Vector Machine (SVM)** classifiers for binary classification. THese models were evaluated and results served as the baseline performance benchmark 

---

### 2. cv_multimodal_fusion.ipynb

This notebook explored the computer vision and multimodal aspects of fake news detection. Visual features were extracted from images using a pretrained **ResNet50** model through transfer learning, producing high-dimensional image embeddings that capture semantic visual information. 
These image representations were then combined with textual feature representations to investigate multimodal learning through feature-level fusion. 
A neural network-based fusion architecture was designed to merge information from both modalities, demonstrating how textual and visual cues can be jointly utilized for misinformation detection.

---

### 3. bert_integration.ipynb

This notebook focused on transformer-based natural language processing using **BERT (Bidirectional Encoder Representations from Transformers)**. News articles were tokenized and converted into contextual embeddings using a pretrained BERT model which also grasps contextual meaning of words unlike TF-IDF representation. The extracted CLS embeddings were used as feature vectors for downstream classification using Logistic Regression. The approach was evaluated on a subset of the fake news dataset and compared against classical NLP baselines to analyze the benefits of contextual language representations.

---

### 4. experiments_and_metrics.ipynb

This notebook shows experimental evaluation and comparison of all developed models. Performance of the classical NLP approaches (TF-IDF + Logistic Regression, Naive Bayes, and SVM) was compared with the BERT-based approach using standard classification metrics including Accuracy, Precision, Recall, F1-Score, and Confusion Matrices.

---

### 5. model_explainability.ipynb

This notebook focused on improving model interpretability using SHAP (SHapley Additive exPlanations). For a feature x_i shap computes average contribution of x_i across all possible feature combinations (A, AB, ABC ...). These are used for quantifying the contribution of each textual feature to the final classification decision. Local explanations were generated to understand why specific articles were classified as fake or real, while global feature importance analysis identified the most influential words affecting model predictions. This enhanced transparency and trustworthiness of the fake news detection pipeline.

---

## Experimental Evaluation

The different approaches were compared using standard classification metrics:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

The experiments provided insights into the trade-offs between traditional machine learning methods, deep learning approaches, and transformer-based language models for fake news detection.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* PyTorch
* TorchVision
* HuggingFace Transformers
* Matplotlib
* Seaborn

---

## Future Work

Planned extensions include:

* Explainability using LIME
* Confidence Calibration
* Uncertainty Estimation
* Advanced Multimodal Fusion Techniques
* End-to-End Multimodal Training

---
