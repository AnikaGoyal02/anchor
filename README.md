# Multimodal Fake News Detection using NLP, Computer Vision and BERT

## Overview

Fake news and online misinformation have become major challenges in the digital age. This project investigates multiple machine learning and deep learning approaches for fake news detection, progressing from classical Natural Language Processing (NLP) techniques to transformer-based language models and multimodal learning architectures.

The project combines textual and visual information to classify news articles as Fake or Real while also improving model interpretability through Explainable AI (XAI) techniques.

The implementation was carried out in multiple phases:

- Classical NLP Baselines using TF-IDF and Machine Learning
- Computer Vision based Feature Extraction using ResNet50
- BERT-based Contextual Language Representations
- Experimental Evaluation and Model Comparison
- Explainability using SHAP
- Attention Based Multimodal Fusion using BERT and ResNet50

The final system demonstrates how textual and visual information can be jointly leveraged for fake news detection through attention-guided multimodal learning.

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

### 6. attention_multimodal_fusion.ipynb

This notebook extends the multimodal fake news detection pipeline through an attention-based fusion architecture using the Fakeddit dataset. Unlike the earlier multimodal approach which relied on direct feature concatenation, this introduces an attention mechanism to determine the relative importance of textual and visual information during classification.

Textual features were extracted using a pretrained BERT Base Uncased model, where each news title was represented using its contextual CLS embedding. Visual features were extracted from associated news images using a pretrained ResNet50 model, producing high-dimensional image embeddings that capture semantic visual content.

The extracted text and image representations were projected into a common feature space and combined through an attention layer. The attention mechanism learned modality-specific weights for text and image features, allowing the model to adaptively emphasize the most informative modality for each news sample. The weighted multimodal representation was then passed through a neural network classifier to predict whether the news item was fake or real.

It also includes data preprocessing, image downloading and validation, feature standardization, model training, performance evaluation using Accuracy, Precision, Recall, F1-Score and Confusion Matrix analysis, along with inspection and visualization of learned attention weights. This experiment demonstrates how attention-guided multimodal learning can be used to combine textual and visual evidence for fake news detection.

Summarising above, this notebook demonstrates how attention mechanisms can be used to effectively combine textual and visual information for multimodal fake news detection.

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
* Fake and Real News Dataset
* Fakeddit Dataset
* SHAP
* ResNet 50

---