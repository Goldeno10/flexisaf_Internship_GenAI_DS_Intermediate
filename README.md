# Flexisaf Internship: Generative AI and Data Science

## Overview
This repository documents my work during the internship at **FlexiSaf**, focusing on Generative AI and Data Science. The internship is structured into modules, each corresponding to a task. I have completed **Task 1**, **Task 2**, **Task 3**, and **Task 4**, as well as a **Final Project**, organized in the directories `task_1`, `task_2`, `task_3`, `task_4`, and `final_project`. All code and notebooks are implemented in Google Colab and Microsoft Scriptbook for accessibility and reproducibility.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Tasks](#tasks)
  - [Task 1: Advanced Machine Learning Techniques](#task-1-advanced-machine-learning-techniques)
  - [Task 2: Deep Learning and Neural Networks](#task-2-deep-learning-and-neural-networks)
  - [Task 3: Natural Language Processing (NLP)](#task-3-natural-language-processing-nlp)
  - [Task 4: Computer Vision](#task-4-computer-vision)
- [Final Project: Admission Success Predictor](#final-project-admission-success-predictor)
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
**Notebooks:**  
- [`task_2/style_transfer.ipynb`](/task_2/style_transfer.ipynb)  
- [`task_2/Character_Level_RNN_Solution.ipynb`](task_2/Character_Level_RNN_Solution.ipynb)

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

### Task 4: Computer Vision
**Module 4: Computer Vision**  
*Duration: 2 Weeks*

In this task, I explored computer vision applications through two distinct projects:

1. **Image Captioning with Flickr8k Dataset**  
   - Built a model that generates natural language descriptions for images using an encoder-decoder architecture (CNN + LSTM).  
   - The encoder extracts visual features from images using a pre-trained CNN (e.g., VGG16/ResNet), while the decoder (LSTM) generates captions conditioned on the image features.  
   - Implemented attention mechanisms to focus on relevant parts of the image during caption generation.  
   - Evaluated using BLEU scores and visualized sample predictions.

2. **Rice Type Classification**  
   - Developed a convolutional neural network (CNN) to classify different varieties of rice (e.g., Arborio, Basmati, Jasmine, etc.) from images.  
   - Applied data augmentation, transfer learning (fine-tuning pre-trained models like ResNet50), and hyperparameter tuning to achieve high accuracy.  
   - The final model demonstrates robust performance on unseen rice images.

Both projects showcase practical computer vision skills, including dataset handling, model architecture design, training pipelines, and evaluation.  
**Technologies:** Python, TensorFlow/Keras, PyTorch, OpenCV, Matplotlib  
**Notebooks:**  
- Image Captioning: [`task_4/image_captioning.ipynb`](task_4/image_captioning.ipynb)  
- Rice Classification: [`task_4/rice_classification.ipynb`](task_4/rice_classification.ipynb)

---

## Final Project: Admission Success Predictor

**🎓 AI-Powered Graduate Guidance**

This project is a machine learning and generative AI application designed to predict a student's chance of getting into graduate school. It uses a Scikit-learn predictive pipeline combined with LangGraph and Google Gemini to offer personalized academic advice.

### 🚀 Overview

Predicting university admission involves many factors. This tool helps students understand their chances based on data such as GRE, TOEFL, and CGPA. It also provides an AI-driven "Counselor" to explain the results and give advice.

### Key Features

- **Predictive Intelligence**: A Random Forest Classifier trained on graduate admission data calculates the probability of success.
- **Automated Pipeline**: A Scikit-learn Pipeline handles data scaling and feature processing.
- **GenAI Counselor**: Powered by LangGraph and Gemini-1.5-Flash, the system creates natural language feedback based on the student's scores.
- **Interactive Dashboard**: Streamlit interface allows real-time predictions and data visualization.

### 🛠️ Tech Stack

- **Data Science**: Python, Pandas, Scikit-learn, Joblib
- **Generative AI**: LangGraph, LangChain, Google Gemini API
- **Frontend**: Streamlit
- **Visualization**: Matplotlib, Seaborn

### 📁 Project Structure

```
final_project/
├── train_model.py      # Automated ML Pipeline & model training
├── agent.py            # LangGraph state management & GenAI logic
├── UI_app.py           # Streamlit dashboard & user interface
├── .env                # API Keys (Protected)
├── .gitignore          # Version control exclusions
└── requirements.txt    # Project dependencies
```

### ⚙️ Getting Started

1. **Installation**  
   Clone the repository and set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

2. **Configuration**  
   Create a `.env` file in the root directory and add your Google API Key:
   ```
   GEMINI_API_KEY=your_actual_key_here
   ```

3. **Usage**  
   First, train the model locally:
   ```bash
   python train_model.py
   ```
   Then launch the application:
   ```bash
   streamlit run UI_app.py
   ```

### 📈 Model Insights

The model assesses features such as CGPA, GRE Scores, and Research Experience. The dashboard includes a Feature Importance chart showing which metrics most affect the individual prediction.

### 🤝 Acknowledgements

This project was developed during the FlexiSAF Edusoft GenAI & Data Science Internship, with a focus on building data-driven tools for the education sector.


## How to Use
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Goldeno10/flexisaf_Internship_GenAI_DS_Intermediate.git 
   cd flexisaf_Internship_GenAI_DS_Intermediate
   ```

2. **Navigate to the desired task folder** (e.g., `cd task_1`).

3. **Open the notebook:**
   - If using Google Colab, click the provided link in the task description, or upload the `.ipynb` file to Colab.
   - Alternatively, run locally with Jupyter: `jupyter notebook task_1.ipynb`

4. **Follow the instructions** inside the notebook to execute cells and explore the results.

All notebooks are self-contained and include necessary markdown explanations.

## Acknowledgments
- FlexiSaf for organizing this internship and providing a structured learning path.
- Microsoft Learn, Udacity, and Kaggle for the high-quality learning resources.
- My mentors and fellow interns for their support and collaboration.