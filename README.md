# Supervised-Text-Classification
Sentiment analysis of Tweets as  “positive”, “negative”, or “neutral”

## Running the program
To train and test classifier for each model, run hw1.py with the following command:

```$ hw1.py --train <path_to_resorces>\data\train.csv```

       --test <path_to_resorces>\data\dev.csv  
       
       --model "Model Name" 
       
       --lexicon_path <path_to_resorces>\lexica\
     

For the purpose of training, I have used the Hashtag-Lexicon file (for unigrams as well as bigrams).
Each of the models have a runtime of:

1. Ngram : 3.84 seconds
2. Ngram+Lex : 6.93 seconds
3. Ngram+Lex+Enc : 7.16 seconds
4. Custom Model : 7.76 seconds

The best performing model is the custom model with an F1 score (macro-averaged) of 0.5077.
Detailed features of each model are given below.

## Model 1: Ngram:

In this model, vanilla ngram model was deployed, using word ngrams from (1,4).The sequence of steps carried out are:

1. The labels with value 'objective' are replaced with 'neutral'.
2. The tweets are lemmatized
3. Training data is fitted on the countvectorizer with ngram range as (1,4)
4. Test data is tranformed on this countvectorizer and all 4 are returned (labels as numpy arrays)
5. Upon training and predicting using SVM (C=10) and Naive Bayes, SVM yielded a result with a higher F1 score. Thus, SVM is chosen.

Here, the features used were:

a. tweet_token : The tweets who's sentiment is to be analyzed, in the form of Ngram vectors from countvectorizer

## Model 2: Ngram+Lex:
In this model, ngram model was deployed, using word ngrams from (1,4). In addition, features are extracted from the hashtag-lexicon file for both unigrams as well as bigrams and are used as additional features to feed into the model. The sequence of steps carried out are:

1. The labels with value 'objective' are replaced with 'neutral'.
2. The tweets are lemmatized
3. All bigrams are extracted from each tweet in order to extract the HS-bigrams in each tweet.
4. The features_unigrams and features_bigrams functions extract features from HS-unigrams and HS-bigrams (features are listed below)
5. Tweets data is fitted on the countvectorizer with ngram range as (1,4)
6. The tweets are then stacked together with the features obtained from step (4).
7. After transforming the test tweets, the same process (6) is repeated for test data and all 4 are returned (labels as numpy arrays)

Here, the features used were:
1. tweet_token : The tweets who's sentiment is to be analyzed, in the form of Ngram vectors from countvectorizer
2. total count of unigram lexicons in the tweet with score(w, p) > 0 where s -> lexicon score
3. total score of all positive unigram lexicons
4. maximal score amongst all positive unigram lexicons
5. score of the last unigram token in the tweet with score(w, p) > 0
6. total count of unigram tokens in the tweet with score(w, p) < 0 where s -> lexicon score
7. total score of all negative unigram lexicons
8. minimum score amongst all negative unigram lexicons
9. score of the last unigram token in the tweet with score(w, p) < 0
10. total count of bigram tokens in the tweet with score(w, p) > 0 where s -> lexicon score
11. total score of all positive bigram lexicons
12. maximal score amongst all positive bigram lexicons
13. score of the last bigram token in the tweet with score(w, p) > 0
14. total count of bigram tokens in the tweet with score(w, p) < 0 where s -> lexicon score
15. total score of all negative bigram lexicons
16. minimum score amongst all negative bigram lexicons
17. score of the last bigram token in the tweet with score(w, p) < 0


## Model 3: Ngram+Lex+Enc:
In this model, ngram model was deployed, using word ngrams from (1,4). In addition, features are extracted from the hashtag-lexicon file for both unigrams as well as bigrams and extracted Encoding features are used as additional features to feed into the model. The sequence of steps carried out are:
1. The labels with value 'objective' are replaced with 'neutral'.
2. The tweets are lemmatized
3. All bigrams are extracted from each tweet in order to extract the HS-bigrams in each tweet.
4. The features_unigrams and features_bigrams functions extract features from HS-unigrams and HS-bigrams (features are listed below)
5. In addition, two features: count of all capitalized words in a tweet, number of hashtags in a tweet are taken
6. Tweets data is fitted on the countvectorizer with ngram range as (1,4)
7. The tweets are then stacked together with the features obtained from steps (4, 5)
8. After transforming the test tweets, the same process (g) is repeated for test data and all 4 are returned (labels as numpy arrays)

Here, the features used were:
1. tweet_token : The tweets who's sentiment is to be analyzed, in the form of Ngram vectors from countvectorizer
2. total count of unigram lexicons in the tweet with score(w, p) > 0 where s -> lexicon score
3. total score of all positive unigram lexicons
4. maximal score amongst all positive unigram lexicons
5. score of the last unigram token in the tweet with score(w, p) > 0
6. total count of unigram tokens in the tweet with score(w, p) < 0 where s -> lexicon score
7. total score of all negative unigram lexicons
8. minimum score amongst all negative unigram lexicons
9. score of the last unigram token in the tweet with score(w, p) < 0
10. total count of bigram tokens in the tweet with score(w, p) > 0 where s -> lexicon score
11. total score of all positive bigram lexicons
l2. maximal score amongst all positive bigram lexicons
13. score of the last bigram token in the tweet with score(w, p) > 0
14. total count of bigram tokens in the tweet with score(w, p) < 0 where s -> lexicon score
15. total score of all negative bigram lexicons
16. minimum score amongst all negative bigram lexicons
17. score of the last bigram token in the tweet with score(w, p) < 0
18. number of words in a tweet with all letters capitalized
19. count of hashtags in a tweet

## Model 4: Custom:
In this model, ngram model was deployed, using word ngrams from (1,4). In addition, features are extracted from the hashtag-lexicon file for both unigrams as well as bigrams and extracted Encoding features are used as additional features to feed into the model. The sequence of steps carried out are:

1. The labels with value 'objective' are replaced with 'neutral'.
2. The tweets are lemmatized
3. All bigrams are extracted from each tweet in order to extract the HS-bigrams in each tweet.
4. The features_unigrams and features_bigrams functions extract features from HS-unigrams and HS-bigrams (features are listed below)
5. In addition, two features: count of all capitalized words in a tweet, number of hashtags in a tweet are taken
6. Tweets data is fitted on the TF-IDF with ngram range as (1,4)
7. The tweets are then stacked together with the features obtained from steps (d,e) and TOP 500 features are selected.
8. After transforming the test tweets, the same process (g) is repeated for test data and all 4 are returned (labels as numpy arrays)

Here, the features used were:
1. tweet_token : The tweets who's sentiment is to be analyzed, in the form of Ngram vectors from TF-IDF
2. total count of unigram lexicons in the tweet with score(w, p) > 0 where s -> lexicon score
3. total score of all positive unigram lexicons
4. maximal score amongst all positive unigram lexicons
5. score of the last unigram token in the tweet with score(w, p) > 0
6. total count of unigram tokens in the tweet with score(w, p) < 0 where s -> lexicon score
7. total score of all negative unigram lexicons
8. minimum score amongst all negative unigram lexicons
9. score of the last unigram token in the tweet with score(w, p) < 0
10. total count of bigram tokens in the tweet with score(w, p) > 0 where s -> lexicon score
11. total score of all positive bigram lexicons
l2. maximal score amongst all positive bigram lexicons
13. score of the last bigram token in the tweet with score(w, p) > 0
14. total count of bigram tokens in the tweet with score(w, p) < 0 where s -> lexicon score
15. total score of all negative bigram lexicons
16. minimum score amongst all negative bigram lexicons
17. score of the last bigram token in the tweet with score(w, p) < 0
18. number of words in a tweet with all letters capitalized
19. count of hashtags in a tweet

Special features of my classifier:
I am using a LinearSVM classifier with a C  value of 10 (for all 4).

While the results are stable, the F1 score was not very high for one class (because of skewed data). Thus, this is a limitation of my classifier. 
