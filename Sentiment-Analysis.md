# Sentiment Analysis

## For IMDB movie reviews

## Team members:

## Chong Xian Jun, Wang Tian, Li Yexin, Ahmed hashim taha salim

# Executive Summary {#executive-summary style="page-break-before:always; "}

## In this study, we conducted a sentiment analysis of movie reviews from IMDB to **predict the overall sentiment of the audience towards a given movie.** We used a combination of natural language processing techniques and machine learning algorithms to analyze the text of the reviews and classify them as either positive or negative.The models used are **logistic regression, multinomial naive bayes, neural network.**

## We found that our model was able to predict the sentiment of the reviews with a high degree of accuracy. Additionally, we also identified some common themes and words that were more prevalent in positive or negative reviews that becomes indicator of the audiences' sentiment.

## Overall, our study provides valuable insights into the sentiment of movie audiences and can be used to inform the marketing and promotion of films.

# Content List {#content-list style="page-break-before:always; "}

## **Introduction and Dataset**

-   Background and Research Questions
-   Introducing the iMDB Movie Review Dataset

## **Methodology**

-   Overview
-   Preprocessing
-   Model Training
-   Model Evaluation
-   Models to Predict Sentiment
-   Important Features

## **Conclusion**

# **Introduction** {#introduction style="page-break-before:always; "}

# Background and Research Questions {#background-and-research-questions style="page-break-before:always; "}

## Conventionally, marketing and promotional campaigns for movies are based on historical data and tend to focus on movies that are similar to previously successful ones. This approach can exclude new or growing markets that might have different preferences.

## To better capture the evolving preferences of movie audiences, we made use of sentiment analysis on IMDB movie review data to build a predictive model. This can be helpful for the stakeholders by automating the extraction of audiences' sentiment on a movie efficiently, thereby informing the promotion and distribution of film.

## The study aims to answer the following questions:

-   How do we predict the overall sentiment of the audiences on a
    particular movie with reviews?
-   What are the important themes that can indicate the audience
    sentiment about a movie?

# IMDB Movie Reviews Dataset {#imdb-movie-reviews-dataset style="page-break-before:always; "}

## Source: [[IMDB Dataset of 50K Movie Reviews \| Kaggle]{.underline}](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

-   Consist of only two columns, "review" and "sentiment".
-   The sentiments are labelled as positive or negative, with each
    sentiment consist of 25000 samples.
-   Contains many syntaxes, symbols and punctuations to be pruned off
    before analysing.

Dataset samples

Dataset info

# **Methodology** {#methodology-1 style="page-break-before:always; "}

# Methodology Overview {#methodology-overview style="page-break-before:always; "}

**Vectorization**

**Model Training and Evaluation**

**Normalization**

**Preprocessing**

Converting text into a standardized representation.

-   Cleaning
-   Tokenization
-   Lemmatization

Converting text into a numerical representation

-   TF-IDF

```{=html}
<!-- -->
```
-   Logistic Regression
-   Neural Network
-   Multinomial Naive Bayes

# Preprocessing {#preprocessing style="page-break-before:always; "}

## Normalization (Cleaning)

## The steps involved in cleaning vary according to data. For the movie reviews, we

-   remove html syntax
-   remove url
-   standardise letter casing
-   fix contractions
-   remove stopwords
-   remove all that are not words

# Before actually cleaning, we write regex function to check if they are properly cleaned. Call function: {#before-actually-cleaning-we-write-regex-function-to-check-if-they-are-properly-cleaned.-call-function style="page-break-before:always; "}

## Before actually cleaning, we write regex function to check if they are properly cleaned. Call function:

## [\`check_if_contain(data, x, pattern)\`]{.underline}

# remove html syntax {#remove-html-syntax style="page-break-before:always; "}

-   remove html syntax
-   Use html-parser from 'BeautifulSoup' to get rid of the html
    syntaxes.

# remove url - remove url with regex pattern starting with https:// or www. {#remove-url---remove-url-with-regex-pattern-starting-with-https-or-www. style="page-break-before:always; "}

-   remove url - remove url with regex pattern starting with https:// or
    www.
-   Standardise letter casing - apply lower() function

# Fix contractions - contractions such as I'm, they're... are expanded with regex {#fix-contractions---contractions-such-as-im-theyre-are-expanded-with-regex style="page-break-before:always; "}

-   Fix contractions - contractions such as I'm, they're... are expanded
    with regex

# Fix contractions (cont) - check if there is still contractions {#fix-contractions-cont---check-if-there-is-still-contractions style="page-break-before:always; "}

-   Fix contractions (cont) - check if there is still contractions

# Remove stopwords - make use of nltk package to remove stop words such as "I'm, myself, the..." {#remove-stopwords---make-use-of-nltk-package-to-remove-stop-words-such-as-im-myself-the style="page-break-before:always; "}

-   Remove stopwords - make use of nltk package to remove stop words
    such as "I'm, myself, the..."

# Remove stopwords - check the first review {#remove-stopwords---check-the-first-review style="page-break-before:always; "}

-   Remove stopwords - check the first review

# Remove all that are not words - use regex to choose only words and white space {#remove-all-that-are-not-words---use-regex-to-choose-only-words-and-white-space style="page-break-before:always; "}

-   Remove all that are not words - use regex to choose only words and
    white space

# Preprocessing {#preprocessing-1 style="page-break-before:always; "}

## Normalization (Tokenization + Lemmatization)

## Tokenization is the process of extracting words as separate tokens. Lemmatization is the process of returning words into its base form, e.g. "plays" -\> "play"

## Tokenization is necessary before lemmatization as it allows the lemmatization algorithm to operate on individual words rather than on the entire text. This makes the lemmatization process more efficient and accurate.

## Several packages from nltk library are used:

# Normalization (Tokenization + Lemmatization) {#normalization-tokenization-lemmatization-1 style="page-break-before:always; "}

## Normalization (Tokenization + Lemmatization)

# Lemmatized review sample compared with that before lemmatization: {#lemmatized-review-sample-compared-with-that-before-lemmatization style="page-break-before:always; "}

## Lemmatized review sample compared with that before lemmatization:

# Preprocessing {#preprocessing-2 style="page-break-before:always; "}

## Vectorization

## Encode the texts into numerical form as before inputting the data into machine learning models. For this study, TF-IDF algorithm is used due to its ability to take into account the frequency and context of words.

## Before vectorization, the normalized dataset is split into X_train, X_test, y_train, y_test. The following is the code for TF-IDF vectorization:

# Vectorized output，X_tfidfvect_train {#vectorized-outputx_tfidfvect_train style="page-break-before:always; "}

## Vectorized output，X_tfidfvect_train

## 

X_tfidfvect_train.shape= (50000, 5000)

# Vectorize y_train and y_test to numerical values as well. {'positive' : 1, 'negative': 0} {#vectorize-y_train-and-y_test-to-numerical-values-as-well.-positive-1-negative-0 style="page-break-before:always; "}

## Vectorize y_train and y_test to numerical values as well. {'positive' : 1, 'negative': 0}

## 

# Model Training and Evaluation {#model-training-and-evaluation style="page-break-before:always; "}

## Models trained are:

-   Logistic Regression
-   Neural Network
-   Multinomial Naive Bayes

# Logistic Regression {#logistic-regression style="page-break-before:always; "}

-   Logistic Regression

# Neural Network {#neural-network style="page-break-before:always; "}

-   Neural Network

# Multinomial Naive Bayes {#multinomial-naive-bayes style="page-break-before:always; "}

-   Multinomial Naive Bayes

# **Results and Findings** {#results-and-findings style="page-break-before:always; "}

# Results and Findings {#results-and-findings-1 style="page-break-before:always; "}

## All in all, we can make use of predictive models to predict audience sentiment based on their review comments. For this purpose, **logistic regression** is the best performing model as it has the highest accuracy with fairly low training and testing time.

  ------------------- --------------------- ------------------------- ----------------
                      Models                                          
                      Logistic Regression   Multinomial Naive Bayes   Neural Network
  Accuracy            0.8864                0.8537                    0.8795
  Training Time (s)   6.706810              0.396690                  509.5316
  Testing Time (s)    0.146578              0.141874                  0.1730
  ------------------- --------------------- ------------------------- ----------------

# Results and Findings {#results-and-findings-2 style="page-break-before:always; "}

## From the logistic regression, we extract the most important features (words) that people associate strongly with positive or negative sentiments.

# **Conclusion** {#conclusion-1 style="page-break-before:always; "}

# Conclusion {#conclusion-2 style="page-break-before:always; "}

## In this study, we conducted a sentiment analysis of movie reviews from IMDB to predict the sentiment of the audience towards a given movie based on unstructured review. It is important to note that thorough preprocessing involving text normalization and vectorization is needed before model training. Our findings showed that logistic regression was able to accurately predict the sentiment of the reviews with accuracy of 0.8864 and fairly efficient training time. he common themes and words that were indicative of positive or negative sentiment are extracted from the model to allow for interpretation.

## These findings have important implications for the film industry as they can be used to inform the marketing and promotion of movies, helping the stakeholders to capture the new market with ever-evolving preferences on movies.
