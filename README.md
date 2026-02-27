# Flexisaf Internship: Generative AI and Data Science

## Overview
This repository documents my work during the internship at **FlexiSaf**, focusing on Generative AI and Data Science. The internship is structured into modules, each corresponding to a task. As of Week 3, I have completed **Task 1**, **Task 2**, and **Task 3**, organized in the directories `task_1`, `task_2`, and `task_3`. All code and notebooks are implemented in Google Colab and microsoft scriptbook for accessibility and reproducibility.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Tasks](#tasks)
  - [Task 1: Advanced Machine Learning Techniques](#task-1-advanced-machine-learning-techniques)
  - [Task 2: Deep Learning and Neural Networks](#task-2-deep-learning-and-neural-networks)
  - [Task 3: Natural Language Processing (NLP)](#task-3-natural-language-processing-nlp)
- [How to Use](#how-to-use)
- [Acknowledgments](#acknowledgments)

## Prerequisites
- Python 3.x
- Basic understanding of machine learning and deep learning concepts
- A Google account to run the notebooks in Colab
- Required libraries (installed automatically in Colab or via `pip`):
  - TensorFlow / PyTorch
  - scikit-learn
  - NLTK / spaCy
  - pandas, numpy, matplotlib
  - (See individual notebooks for full list)

## Tasks

### Task 1: Advanced Machine Learning Techniques
**Module 1: Advanced Machine Learning Techniques**  
*Duration: 2 Weeks*

In this task, I explored multiple advanced machine learning techniques and demonstrated practical understanding by training models on at least two of the following:
- Deep Learning (neural networks with multiple layers)
- Reinforcement Learning (agent-based learning)
- Ensemble Learning (Random Forests, Gradient Boosting)
- Transfer Learning (fine-tuning pre-trained models)
- Semi-Supervised and Self-Supervised Learning
- Generative Adversarial Networks (GANs)
- AutoML
- Hyperparameter Optimization
- Time Series Forecasting with RNNs
- Meta-Learning (e.g., MAML)
- Explainable AI (LIME, SHAP)

The notebook includes implementations, explanations, and performance comparisons for the chosen techniques, highlighting their real-world applications.  
**Technologies:** Python, scikit-learn, TensorFlow/Keras, PyTorch, XGBoost, etc.  
**Notebook:** [`task_1`](task_1/)

### Task 2: Deep Learning and Neural Networks
**Module 2: Deep Learning and Neural Networks**  
*Duration: 2 Weeks*

This task involved completing the **Udacity course "Deep Learning with PyTorch"** to build a strong foundation in neural networks. Key concepts covered:
- Neurons, layers, weights, and biases
- Activation functions (sigmoid, tanh, ReLU)
- Feedforward and backpropagation
- Deep Neural Networks (DNN), CNNs, RNNs, LSTMs, GRUs, Autoencoders

A screenshot of the course completion (PDF) is included in the task directory as proof. The notebook also contains practical exercises and implementations using PyTorch.  
**Technologies:** PyTorch, Python, Jupyter/Colab  
**Completion Proof:** [`task_2/course_completion.pdf`](task_2/flexisaf_internship_genai_and_ds_intermediate_task_2.pdf)
**Notebook:** [`task_2/style_transfer.ipynb`](/task_2/style_transfer.ipynb)
**Notebook:** [`task_2/Character_Level_RNN_Solution.ipynb`](task_2/Character_Level_RNN_Solution.ipynb)

### Task 3: Natural Language Processing (NLP)
**Module 3: Natural Language Processing**  
*Duration: 2 Weeks*

This task focuses on essential text preprocessing techniques for NLP. Using sample data from Kaggle (or any other source), I applied at least three of the following common text cleaning steps:
- Lower casing
- Removal of punctuation
- Removal of frequent words
- Removal of rare words
- Stemming
- Lemmatization
- Removal of emojis / emoticons
- Conversion of emojis / emoticons to words
- Removal of URLs
- Spelling correction

The notebook demonstrates each preprocessing step, shows before-and-after examples, and discusses the impact on downstream tasks such as sentiment analysis or text classification.  
**Technologies:** Python, NLTK, spaCy, regex, pandas  
**Notebook:** [`task_3/task_3_Text_Preprocessing.ipynb`](task_3/task_3_Text_Preprocessing.ipynb)

## How to Use
1. **Clone the repository:**
   ```bash
   git clone  https://github.com/Goldeno10/flexisaf_Internship_GenAI_DS_Intermediate.git 
   cd flexisaf_Internship_GenAI_DS_Intermediate

2. **Navigate to the desired task folder** (e.g., `cd task_1`).

3. **Open the notebook:**
- If using Google Colab, click the provided link in the task description, or upload the `.ipynb` file to Colab.
- Alternatively, run locally with Jupyter: `jupyter notebook task_1.ipynb`

Follow the instructions inside the notebook to execute cells and explore the results.


## Acknowledgments
- FlexiSaf for organizing this internship and providing a structured learning path.
- Microsoft Learn, Udacity, and Kaggle for the high-quality learning resources.
- My mentors and fellow interns for their support and collaboration.