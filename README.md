
# Sentiment Analysis For IMDB movie reviews

# Executive Summary

- In this study, we conducted a sentiment analysis of movie 
reviews from IMDB to **predict the overall sentiment of the 
audience towards a given movie.** We used a combination of 
natural language processing techniques and machine learning
algorithms to analyze the text of the reviews and classify 
them as either positive or negative.The models used are 
**logistic regression, multinomial naive bayes, neural network.**

<div align="justify">
We found that our model was able to predict the sentiment of the 
reviews with a high degree of accuracy. Additionally, we also 
identified some common themes and words that were more prevalent 
in positive or negative reviews that becomes indicator of the 
audiences' sentiment.
</div>
<br>
    
<div align="justify">
Overall, our study provides valuable insights into the sentiment
of movie audiences and can be used to inform the marketing and 
promotion of films.
</div>


# Background and Research Questions

<div align="justify">
    Conventionally, marketing and promotional campaigns for movies are
    based on historical data and tend to focus on movies that are
    similar to previously successful ones. This approach can exclude
    new or growing markets that might have different preferences.
</div>
<br>
<div align="justify">
    To better capture the evolving preferences of movie audiences,
    we made use of sentiment analysis on IMDB movie review data to 
    build a predictive model. This can be helpful for the stakeholders
    by automating the extraction of audiences' sentiment on a movie 
    efficiently, thereby informing the promotion and distribution of film.
        </div><br>
    The study aims to answer the following questions:

-   How do we predict the overall sentiment of the audiences on a
    particular movie with reviews?
-   What are the important themes that can indicate the audience
    sentiment about a movie?

# IMDB Movie Reviews Dataset 

Source: [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

-   Consist of only two columns, "review" and "sentiment".
-   The sentiments are labelled as positive or negative, with each
    sentiment consist of 25000 samples.
-   Contains many syntaxes, symbols and punctuations to be pruned off
    before analysing.

![Picture1](https://user-images.githubusercontent.com/126220185/222567838-3e1f5ed9-fcf2-40a1-9ab7-83049c216c27.png)
![Picture2](https://user-images.githubusercontent.com/126220185/222567855-567778c0-c976-4401-93a7-eff0d675658d.png)




# Methodology

## 1. Preprocessing <br>

i.  Normalization <br>
Converting text into a standardized representation, including:
-   Cleaning
-   Tokenization
-   Lemmatization

ii- Vectorization <br>
Converting text into a numerical representation, including:
-   TF-IDF 
-   BOW "Not Used"

## 2. Model Training and Evaluation
-   Logistic Regression
-   Neural Network
-   Multinomial Naive Bayes





## 1. Preprocessing 

Normalization (Cleaning) <br>

The steps involved in cleaning vary according to data. For the movie reviews, we

-   remove html syntax
-   remove url
-   standardise letter casing
-   fix contractions
-   remove stopwords
-   remove all that are not words



## 2. Normalization (Tokenization + Lemmatization) <br>

Tokenization is the process of extracting words as separate tokens.
Lemmatization is the process of returning words into its base form, 
e.g. "plays" -\> "play" 

<br>

Tokenization is necessary before lemmatization as it allows the 
lemmatization algorithm to operate on individual words rather than
on the entire text. This makes the lemmatization process more efficient
and accurate.




## 3. Vectorization

Encode the texts into numerical form as before inputting the data 
into machine learning models. For this study, TF-IDF algorithm is used 
due to its ability to take into account the frequency and context of words.

<br>

Before vectorization, the normalized dataset is split
into X_train, X_test, y_train, y_test.
The following is the code for TF-IDF vectorization:


## 4. Model Training and Evaluation 

Models trained are:
-   Logistic Regression
-   Neural Network
-   Multinomial Naive Bayes


## Results and Findings

<div align="justify">
All in all, we can make use of predictive models to predict
audience sentiment based on their review comments. For this purpose, 
**logistic regression** is the best performing model as it has the 
highest accuracy with fairly low training and testing time.
</div>

  ------------------- --------------------- ------------------------- ----------------
                      Models                                          
                      Logistic Regression   Multinomial Naive Bayes   Neural Network
  Accuracy            0.8864                0.8537                    0.8795
  Training Time (s)   6.706810              0.396690                  509.5316
  Testing Time (s)    0.146578              0.141874                  0.1730
  ------------------- --------------------- ------------------------- ----------------


## Conclusion 

<div align="justify">
In this study, we conducted a sentiment analysis of movie reviews from
IMDB to predict the sentiment of the audience towards a given movie based on
unstructured review. It is important to note that thorough preprocessing involving
text normalization and vectorization is needed before model training. Our findings
showed that logistic regression was able to accurately predict the sentiment of the
reviews with accuracy of 0.8864 and fairly efficient training time. he common themes
and words that were indicative of positive or negative sentiment are extracted from
the model to allow for interpretation.
</div> <br>

<div align="justify">
These findings have important implications for the film industry as they can be used
to inform the marketing and promotion of movies, helping the stakeholders to capture
the new market with ever-evolving preferences on movies.
</div>
