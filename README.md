# Darwin Handwriting Classification Project

This project utilizes the DARWIN dataset to classify Alzheimer's Disease (AD) based on handwriting features. The goal is to analyze handwriting tasks performed by participants, both healthy and diagnosed with AD, and predict AD based on distinct cognitive and motor patterns extracted from their handwriting.

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset Overview](#dataset-overview)
3. [Data Preprocessing](#data-preprocessing)
4. [Feature Selection](#feature-selection)
5. [Model Training](#model-training)
6. [Model Evaluation](#model-evaluation)
7. [Conclusion](#conclusion)

---

## Introduction

This notebook is designed to provide insights into diagnosing Alzheimer's Disease from handwriting samples. The dataset used in this project comes from the 2022 study by Cilia et al., which offers a unique set of handwriting tasks and features relevant to AD detection. This notebook focuses on training classification models to predict the disease.

---

## Dataset Overview

The DARWIN dataset consists of 174 participants, including 89 individuals diagnosed with Alzheimer’s Disease and 85 healthy controls. Participants completed 25 handwriting tasks, and for each task, 18 features were extracted, such as air time, pressure, jerk, and tremor.

### Key Features:
- **ID**: Participant identifier
- **Class**: Label indicating whether the participant is healthy (0) or has Alzheimer’s Disease (1)
- **Total time**: Time taken to complete each task
- **Air time**: Duration of pen lift during handwriting
- **Pressure**: Pressure applied during writing
- **Jerk**: A measure of the smoothness of pen movement
- **Tremor**: Shakiness in pen movement

---

## Data Preprocessing

Preprocessing steps:

1. **Handling Missing Values**: Any missing or incomplete data is addressed.
2. **Feature Selection**: A subset of tasks and features is chosen based on performance metrics.
3. **Normalization**: Features are scaled to ensure equal contribution to the model.
4. **Train-Test Split**: The data is split into training and validation sets.

---

## Feature Selection

For this analysis, I’ve chosen a subset of handwriting tasks that performed well across multiple models, as reported in the original DARWIN study. These tasks include:

- Task 5, Task 6, Task 7, Task 15, Task 16, Task 17, Task 19, Task 23, Task 24

The goal is to focus on the most influential tasks, helping improve the model's generalizability and reduce bias.

---

## Model Training

### Models Used:
- **Logistic Regression**: A simple yet powerful linear model.
- **Random Forest**: A great method that uses decision trees.
- **Support Vector Machine (SVM)**: A classifier that separates data with the largest margin.

Hyperparameter tuning is done to optimize each model, focusing on improving accuracy, recall, and F1 scores.

---

## Model Evaluation

After training the models, they are evaluated using metrics such as accuracy, recall, F1 score, and confusion matrices. The goal is to assess the models' ability to correctly predict Alzheimer's Disease (AD) while minimizing false positives and false negatives.

---

## Conclusion

This notebook demonstrates the application of machine learning models to classify Alzheimer's Disease from handwriting data. The Random Forest model performed the best, making it a strong candidate for further refinement and deployment in real-world applications.

### Future Work:
- Further tuning of hyperparameters in SVM for better performance
- Exploration of more advanced models (ex: XGBoost, Neural Networks)
- Incorporating additional handwriting features for improved accuracy

---

## Requirements

- Python 
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`
- Dataset: DARWIN dataset (please check the original paper for access details)
  Cilia, N. D., De Gregorio, G., De Stefano, C., Fontanella, F., Marcelli, A., & Parziale, A. (2022). Diagnosing Alzheimer’s disease from on-line handwriting: A novel dataset and performance benchmarking. Engineering Applications of Artificial Intelligence, 116, 104822. https://doi.org/10.1016/j.engappai.2022.104822

