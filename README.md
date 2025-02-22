# Sentiment Analysis with Naïve Bayes Classifier using Apache Spark

##  Overview

This project focuses on big data analysis by implementing a **Naïve Bayes classifier** using **Apache Spark**. The main objective is to classify user reviews from **Amazon** based on sentiment analysis and **Natural Language Processing (NLP)** techniques.

##  Description

The dataset consists of customer reviews from Amazon, where we preprocess textual data and apply **sentiment classification** to predict user ratings. The implementation is written in **Python** and leverages **Apache Spark MLlib** for scalable machine learning.

##  Features

- **Big Data Processing**: Efficient handling of large-scale datasets with **Apache Spark**.
- **NLP Techniques**: Text preprocessing, tokenization, and feature extraction.
- **Naïve Bayes Classifier**: A supervised learning model to predict user ratings.
- **Sentiment Analysis**: Understanding user opinions through classification.
- **Scalability**: Optimized for execution in a distributed environment.

##  Technologies Used

- **Apache Spark**
- **Python (PySpark, NLP libraries)**
- **Machine Learning (Naïve Bayes Classification)**
- **Jupyter Notebook (for analysis and visualization)**

##  Installation & Setup

### Prerequisites

Ensure you have the following installed:

- Python 3.x
- Apache Spark
- Jupyter Notebook (optional, for interactive exploration)
- Required Python libraries:
  ```sh
  pip install pyspark nltk pandas numpy
  ```

### Running the Project

1. Clone the repository:
   ```sh
   git clone https://github.com/your-repository/sentiment-analysis-spark.git
   cd sentiment-analysis-spark
   ```
2. Start Apache Spark:
   ```sh
   pyspark
   ```
3. Run the Python script using the following command:
   ```sh
   spark-submit python_code.py

   ```

##  Dataset

The dataset consists of Amazon user reviews. Preprocessing steps include:

- Removing stop words
- Tokenization
- Vectorization using TF-IDF

##  Code Overview

The **Python script** performs the following tasks:
- Loads and preprocesses the dataset from HDFS
- Cleans and tokenizes the text data
- Applies stemming and lemmatization
- Converts text data into numerical features using TF-IDF
- Splits the dataset into training and testing sets with different ratios
- Trains a **Naïve Bayes** model using Apache Spark MLlib
- Evaluates the model performance using precision and recall


