# British Airways Sentiment Analysis and Customer Booking Prediction

This project involves web scraping and analysis of reviews for **British Airways** from the **Skytrax** website. It utilizes **Python** libraries such as **BeautifulSoup**, **NLTK**, and **VADER** for sentiment analysis of customer reviews. Additionally, the project incorporates **predictive modeling** to predict customer booking behavior using a **Random Forest Classifier** and **XGBoost** classifier based on customer booking data.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Requirements](#requirements)
3. [Data Collection and Web Scraping](#data-collection-and-web-scraping)
4. [Sentiment Analysis](#sentiment-analysis)
5. [Customer Booking Behavior Prediction](#customer-booking-behavior-prediction)
6. [Visualizations](#visualizations)
7. [Model Performance](#model-performance)
8. [Conclusion](#conclusion)

## Project Overview


This project focuses on:
1. **Sentiment Analysis**: Scraping reviews from the **British Airways** page on **Skytrax**, cleaning the text, performing tokenization, part-of-speech (POS) tagging, lemmatization, and analyzing sentiments using the **VADER Sentiment Analyzer**.
2. **Customer Booking Behavior Prediction**: Using customer booking data to predict if a booking will be completed based on various features such as `route`, `booking_origin`, and `flight_duration`.

The analysis uses several machine learning models to classify customer booking behavior, comparing their performance and selecting the best model.

## Requirements

To run this project, you need the following Python packages:
- `requests`
- `beautifulsoup4`
- `pandas`
- `nltk`
- `matplotlib`
- `seaborn`
- `wordcloud`
- `vaderSentiment`
- `xgboost`
- `scikit-learn`
- `chardet`

## Data Collection and Web Scraping

The review data for **British Airways** was collected using **BeautifulSoup** from the **Skytrax** website ([link](https://www.airlinequality.com/airline-reviews/british-airways)). The scraper fetches reviews from multiple pages, and each review's content is extracted and saved into a `.csv` file.

### Key Steps:
1. Scraping review links from the website.
2. Extracting review content.
3. Storing the reviews in a CSV file for further processing.

## Sentiment Analysis

Using the **VADER Sentiment Analyzer**, the reviews are analyzed to categorize them into **positive**, **negative**, and **neutral** sentiments.

### Data Preprocessing:
- Cleaning the text data.
- Tokenizing and removing stop words.
- Part-of-speech tagging and lemmatization.

### Sentiment Categorization:
Sentiments are classified based on the **compound** score from VADER:
- Positive: Compound score >= 0.5
- Negative: Compound score < 0
- Neutral: Compound score between 0 and 0.5

## Customer Booking Behavior Prediction

The customer booking data (from `customer_booking.csv`) is used to predict whether a booking will be completed based on various features.

### Key Steps:
1. **Feature Selection**: Identifying the most important features using **Mutual Information** scores.
2. **Model Training**: Using **Random Forest Classifier** and **XGBoost Classifier** for classification.
3. **Model Evaluation**: Evaluating the models using **accuracy** and **AUC score**.

## Visualizations

The project includes various visualizations, such as:
- **Pie charts** for sentiment distribution.
![image](https://github.com/user-attachments/assets/6e41d5da-6500-4d76-b180-a8ea427ffcde)

- **Bar charts** for sentiment count.
![image](https://github.com/user-attachments/assets/f079c4cb-5cc3-4940-b344-086e13fd0395)

- **Word clouds** for positive, negative, and neutral reviews.
![image](https://github.com/user-attachments/assets/7fc1d3ca-abe4-4447-b52a-1dc8bbbcd560)

![image](https://github.com/user-attachments/assets/4b3a3dd7-abb9-40f8-84bc-6c061f9e9480)

![image](https://github.com/user-attachments/assets/0f173b8d-edba-4a1e-aeb5-542489c3d05e)


## Model Performance

Several models were evaluated:
1. **Random Forest Classifier** using top 6 features and all features.
2. **XGBoost Classifier** using top 6 features and all features.

The best-performing model was **Random Forest Classifier with all features**, achieving a good accuracy of 85% and AUC score of 0.558




