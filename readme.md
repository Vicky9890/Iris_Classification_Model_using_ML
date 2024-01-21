# Iris Classification using Machine Learning

The "Iris Classification using Machine Learning" script is a Python program designed to classify Iris flowers into different species based on their physical characteristics. The script employs machine learning techniques, specifically a neural network implemented using the TensorFlow and Keras libraries, to train and evaluate a model on the well-known Iris dataset.

## Description

This project utilizes the popular Iris dataset, which contains measurements of sepal length, sepal width, petal length, and petal width for three different species of Iris flowers. The script employs various Python libraries, including pandas, numpy, seaborn, tensorflow, and scikit-learn.

## Dataset

The dataset used in this project is the famous Iris dataset, which consists of measurements of sepal length, sepal width, petal length, and petal width for three different species of Iris flowers.

- SepalLengthCm
- SepalWidthCm
- PetalLengthCm
- PetalWidthCm
- Species

## Data Preprocessing

The model performs the following data preprocessing steps:

- **Loading Data:** Reads the dataset using pandas.

- **Exploratory Data Analysis (EDA):** Visualizes the data using seaborn and matplotlib to gain insights.

- **Label Encoding:** Converts the categorical labels (Species) into numerical format using LabelEncoder.

- **Data Splitting:** Splits the dataset into training and testing sets.

## Requirements 

Ensure you have the following dependencies installed before running the model:

- pandas
- numpy
- seaborn
- tensorflow
- matplotlib
- scikit-learn

You can install these requirements using the following command:

```bash
pip install pandas numpy seaborn tensorflow matplotlib scikit-learn
```

## Usage

To use the Movie Recommendation System, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/Vicky9890/Iris_Classification_Model_using_ML.git
```

2. Navigate to the project directory:
```bash
cd Iris Classification
```

3. Run the Classification model:
```bash
jupyter-notebook Iris_Classification.ipynb
```

## Model Architecture

The classification model is constructed using a Sequential neural network from Keras. The architecture includes:

- **Input Layer:** Dense layer with input dimension matching the number of features (4 in this model).

- **Hidden Layers:** Dense layers with ReLU activation functions.

- **Output Layer:** Dense layer with softmax activation function for multi-class classification.

## Model Evaluation

After training, the model is evaluated on the test set, and the following metrics are provided:

- Accuracy Score
- Confusion Matrix
- R-squared Score
- Mean Absolute Error
- Mean Squared Error