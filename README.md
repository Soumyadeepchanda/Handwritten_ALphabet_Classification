# Handwritten Alphabet Classification using Convolutional Neural Networks

This project aims to recognize handwritten alphabets using Convolutional Neural Networks (CNNs). The dataset used for training and testing is the "A_Z Handwritten Data" dataset. The trained model achieves an accuracy of 98.62% in classifying the handwritten alphabets.

## Table of Contents
1. [Import Dependencies and Setup](#import-dependencies-and-setup)
2. [Data Loading and Exploration](#data-loading-and-exploration)
3. [Data Cleaning](#data-cleaning)
4. [Exploratory Data Analysis](#exploratory-data-analysis)
5. [Data Preprocessing](#data-preprocessing)
6. [Convolutional Neural Network (CNN) Model Build](#convolutional-neural-network-cnn-model-build)
7. [Model Training](#model-training)
8. [Model Evaluation](#model-evaluation)
9. [Filters Visualization](#filters-visualization)
10. [Classification Report](#classification-report)
11. [Confusion Matrix Visualization](#confusion-matrix-visualization)

## Import Dependencies and Setup
In this section, the necessary libraries such as pandas, numpy, os, matplotlib, tensorflow, keras, seaborn, and PIL are imported. Any additional setup or configurations required for the project are also included.

## Data Loading and Exploration
The dataset is loaded using pandas' `read_csv()` function, and various exploratory analysis functions like `head()`, `shape`, `info()`, `dtypes`, `describe()`, `nunique()`, and `columns` are used to gain insights into the loaded data.

## Data Cleaning
Duplicate records are checked and removed using pandas' `drop_duplicates()` function. Null values, if any, are identified and handled accordingly.

## Exploratory Data Analysis
The count of each alphabet in the dataset is visualized using a bar plot to understand the distribution of data.

## Data Preprocessing
The pixel values of the images are normalized by dividing them by 255. The dataset is split into training and testing sets using the `train_test_split()` function from sklearn or any other method used for data preprocessing.

## Convolutional Neural Network (CNN) Model Build
A sequential model is created using Keras' `Sequential()` class. Conv2D layers with ReLU activation, MaxPooling2D layers, and a Flatten layer are added to the model. Dense layers with ReLU activation and a final Dense layer with softmax activation are also added for classification. The model is compiled using the Adam optimizer and sparse categorical cross-entropy loss.

## Model Training
The model is trained on the training data using the `fit()` function. The number of epochs, batch size, and validation data are specified during the training process.

## Model Evaluation
The performance of the model is evaluated on the test data using the `evaluate()` function. Metrics such as accuracy, precision, recall, and F1 score are calculated to assess the model's performance.

## Filters Visualization
The learned filters of the first Conv2D layer are plotted to visualize the feature extraction process of the model.

## Classification Report
A classification report is generated, which includes precision, recall, F1 score, and support for each class. This report provides detailed insights into the model's performance for individual classes.

## Confusion Matrix Visualization
A confusion matrix is generated to visualize the model's performance on the test data. It helps in understanding the number of correct and incorrect predictions made by the model.
