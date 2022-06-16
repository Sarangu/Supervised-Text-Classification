#Import necessary libraries
import pandas as pd
import numpy as np
import string
import re
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
from nltk.tokenize import RegexpTokenizer
from scipy import sparse
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest, chi2

#Creating class features
class features:
    def __init__(self, train_file, test_file, model, lexicon_path):
        self.train = pd.read_csv(train_file)#Reading training file into dataframe
        self.test = pd.read_csv(test_file)#Reading testing file into dataframe
        self.model = model
        self.lexicon_path = pd.read_csv(lexicon_path+'Hashtag-Sentiment-Lexicon\HS-unigrams.txt', sep='\t', header = None)#Reading Unigram lexicon file into dataframe
        self.lexicon_path.columns = ['lexicon', 'sent_score', 'N_pos', 'N_neg']#Assigning columns for the Unigram lexicon data
        self.lexicon_path_bigrams = pd.read_csv(lexicon_path+'Hashtag-Sentiment-Lexicon\HS-bigrams.txt', sep='\t', header = None)#Reading Bigram lexicon file into dataframe
        self.lexicon_path_bigrams.columns = ['bigram', 'sent_score', 'N_pos', 'N_neg']#Assigning columns for Bigram Lexicon data
        self.lex_dict = dict(zip(self.lexicon_path.lexicon, self.lexicon_path.sent_score))#Creating Unigram Lexicon Dictionary to improve speed of search
        self.lex_dict_bigram = dict(zip(self.lexicon_path_bigrams.bigram, self.lexicon_path_bigrams.sent_score))#Creating Bigram Lexicon Dictionary to improve speed of search
    #Function to remove handles in a tweet (starting with @)    
    def rem_handles(self, tweet_corpus):
        return " ".join(filter(lambda x:x[0]!='@', tweet_corpus.split()))
    #Function to lemmatize tweets
    def lemmatize(self, tweet_corpus):
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        return [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in tweet_corpus]
    
    #Function to obtain unigram features from lexicons - Total score, count, max_score/min_score, last score for lexicons with both positive and negative polarity
    def features_unigrams(self, column):
        n = len(column)
        count = np.zeros(n)
        total = np.zeros(n)
        max_score = np.zeros(n)
        last_score = np.zeros(n)
        count_neg = np.zeros(n)
        total_neg = np.zeros(n)
        min_score = np.zeros(n)
        last_score_neg = np.zeros(n)
        for i in range(n):
            for word in column[i].split():
                if word in self.lex_dict.keys():
                    score = self.lex_dict[word]
                    if score > 0:
                        count[i] = count[i] + 1
                        total[i] = total[i] + score
                        max_score[i] = max(max_score[i], score)
                        last_score[i] = score
                    if score < 0:
                        count_neg[i] = count_neg[i] + 1
                        total_neg[i] = total_neg[i] - score
                        min_score[i] = max(min_score[i], (-1 * score))
                        last_score_neg[i] = abs(score)
        return (count, total, max_score, last_score, count_neg, total_neg, min_score, last_score_neg)
    
    #Function to obtain bigram features from lexicons - Total score, count, max_score/min_score, last score for lexicons with both positive and negative polarity
    def features_bigrams(self, column):
        n = len(column)
        count = np.zeros(n)
        total = np.zeros(n)
        max_score = np.zeros(n)
        last_score = np.zeros(n)
        count_neg = np.zeros(n)
        total_neg = np.zeros(n)
        min_score = np.zeros(n)
        last_score_neg = np.zeros(n)
        for i in range(n):
            for bigram_unit in column[i]:
                if bigram_unit in self.lex_dict_bigram.keys():
                    score = self.lex_dict_bigram[bigram_unit]
                    if score > 0:
                        count[i] = count[i] + 1
                        total[i] = total[i] + score
                        max_score[i] = max(max_score[i], score)
                        last_score[i] = score
                    if score < 0:
                        count_neg[i] = count_neg[i] + 1
                        total_neg[i] = total_neg[i] - score
                        min_score[i] = max(min_score[i], (-1 * score))
                        last_score_neg[i] = abs(score)
        return (count, total, max_score, last_score, count_neg, total_neg, min_score, last_score_neg)
    #Function for extracting no. of words with all upper case letters
    def no_upper_case(self, text):
        count = 0
        for word in text.split():
            if (word.isupper()):
                count = count + 1
        return count
    #Function for extracting number of hashtags per tweet
    def hashtag_counter(self, text):
        count = text.count('#')
        return count
    #Model1:Function that will return X_train, Y_train, X_test, Y_test if model is given to be Ngram
    def Ngram(self):
        self.train.loc[self.train['label'] == "objective", 'label'] = "neutral"
        self.test.loc[self.test['label'] == "objective", 'label'] = "neutral"
        self.train['tweet_tokens'] = self.train['tweet_tokens'].apply(lambda x: features.rem_handles(self, x))
        self.train['tweet_tokens'] = self.train['tweet_tokens'].str.replace("[^a-zA-Z#?!.,]", " ")
        tweet_corpus = self.train['tweet_tokens']
        tweet_corpus = features.lemmatize(self, tweet_corpus)
        X = np.array(tweet_corpus)
        Y = np.array(self.train['label'].values.tolist())
        token = RegexpTokenizer(r'[a-zA-Z0-9]+')
        cv = CountVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 4), tokenizer=token.tokenize)#countvectorizer
        X = cv.fit_transform(X.ravel())
        X_test = cv.transform(self.test['tweet_tokens'])
        Y_test = np.array(self.test['label'].values.tolist())
        return (X, Y, X_test, Y_test)
    #Model2:Function that will return X_train, Y_train, X_test, Y_test if model is given to be Ngram_Lex
    def Ngram_Lex(self):
        self.train.loc[self.train['label'] == "objective", 'label'] = "neutral"
        self.test.loc[self.test['label'] == "objective", 'label'] = "neutral"
        tweet_corpus = self.train['tweet_tokens']
        tweet_corpus = features.lemmatize(self, tweet_corpus)
        X = np.array(tweet_corpus)
        Y = np.array(self.train['label'].values.tolist())
        token = RegexpTokenizer(r'[a-zA-Z0-9]+')
        cv = CountVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 4), tokenizer=token.tokenize)#countvectorizer
        X = cv.fit_transform(X.ravel())
        
        #obtaining bigrams from tweets
        train_bigram_list = []
        test_bigram_list = []
        vectorizer = CountVectorizer(ngram_range = (2,2))
        for i in range(len(self.train["tweet_tokens"])):
            vectorizer.fit([self.train.iloc[i]["tweet_tokens"]])
            train_bigram_list.append(vectorizer.get_feature_names())
        for i in range(len(self.test["tweet_tokens"])):
            vectorizer.fit([self.test.iloc[i]["tweet_tokens"]])
            test_bigram_list.append(vectorizer.get_feature_names())        
        self.train['bigrams'] = train_bigram_list
        self.test['bigrams'] = test_bigram_list
        
        #creating columns for lexicon based features for training data
        self.train['positive_score_count'], self.train['Total_pos_score'], self.train['max_score'], self.train['last_pscore'], self.train['negative_score_count'], self.train['Total_neg_score'], self.train['min_score'], self.train['last_nscore'] = features.features_unigrams(self, self.train['tweet_tokens'])
        
        self.train['bigram_positive_score_count'], self.train['bigram_Total_pos_score'], self.train['bigram_max_score'], self.train['bigram_last_pscore'], self.train['bigram_negative_score_count'], self.train['bigram_Total_neg_score'], self.train['bigram_min_score'], self.train['bigram_last_nscore'] = features.features_bigrams(self, self.train['bigrams'])
                
        train_features = np.array([self.train.positive_score_count, self.train.Total_pos_score, self.train.max_score, self.train.last_pscore, self.train.negative_score_count, self.train.Total_neg_score, self.train.min_score, self.train.last_nscore, self.train.bigram_positive_score_count, self.train.bigram_Total_pos_score, self.train.bigram_max_score, self.train.bigram_last_pscore, self.train.bigram_negative_score_count, self.train.bigram_Total_neg_score, self.train.bigram_min_score, self.train.bigram_last_nscore])

        train_features = train_features.transpose()
        train_features = sparse.coo_matrix(train_features)
        X = hstack((X, train_features.astype(float)))  #stacking ngram with other features (Lex+Enc)
        
        X_test = cv.transform(self.test['tweet_tokens'])
        Y_test = np.array(self.test['label'].values.tolist())
        
        #creating columns for lexicon based features for test data
        self.test['positive_score_count'], self.test['Total_pos_score'], self.test['max_score'], self.test['last_pscore'], self.test['negative_score_count'], self.test['Total_neg_score'], self.test['min_score'], self.test['last_nscore'] = features.features_unigrams(self, self.test['tweet_tokens'])
        
        self.test['bigram_positive_score_count'], self.test['bigram_Total_pos_score'], self.test['bigram_max_score'], self.test['bigram_last_pscore'], self.test['bigram_negative_score_count'], self.test['bigram_Total_neg_score'], self.test['bigram_min_score'], self.test['bigram_last_nscore'] = features.features_bigrams(self, self.test['bigrams'])
                
        test_features = np.array([self.test.positive_score_count, self.test.Total_pos_score, self.test.max_score, self.test.last_pscore, self.test.negative_score_count, self.test.Total_neg_score, self.test.min_score, self.test.last_nscore, self.test.bigram_positive_score_count, self.test.bigram_Total_pos_score, self.test.bigram_max_score, self.test.bigram_last_pscore, self.test.bigram_negative_score_count, self.test.bigram_Total_neg_score, self.test.bigram_min_score, self.test.bigram_last_nscore])

        test_features = test_features.transpose()
        test_features = sparse.coo_matrix(test_features)
        X_test = hstack((X_test, test_features.astype(float)))  #stacking ngram with other features (Lex+Enc)
        return(X, Y, X_test, Y_test)
    
    #Model3:Function that will return X_train, Y_train, X_test, Y_test if model is given to be Ngram_Lex_Enc
    def Ngram_Lex_Enc(self):
        self.train.loc[self.train['label'] == "objective", 'label'] = "neutral"
        self.test.loc[self.test['label'] == "objective", 'label'] = "neutral"
        tweet_corpus = self.train['tweet_tokens']
        tweet_corpus = features.lemmatize(self, tweet_corpus)
        X = np.array(tweet_corpus)
        Y = np.array(self.train['label'].values.tolist())
        token = RegexpTokenizer(r'[a-zA-Z0-9]+')
        cv = CountVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 4), tokenizer=token.tokenize)#countvectorizer
        X = cv.fit_transform(X.ravel())
        
        #obtaining bigrams from tweets   
        train_bigram_list = []
        test_bigram_list = []
        vectorizer = CountVectorizer(ngram_range = (2,2))
        for i in range(len(self.train["tweet_tokens"])):
            vectorizer.fit([self.train.iloc[i]["tweet_tokens"]])
            train_bigram_list.append(vectorizer.get_feature_names())
        for i in range(len(self.test["tweet_tokens"])):
            vectorizer.fit([self.test.iloc[i]["tweet_tokens"]])
            test_bigram_list.append(vectorizer.get_feature_names())
            
        self.train['bigrams'] = train_bigram_list
        self.test['bigrams'] = test_bigram_list
        
        #creating columns for lexicon based features for training data
        self.train['positive_score_count'], self.train['Total_pos_score'], self.train['max_score'], self.train['last_pscore'], self.train['negative_score_count'], self.train['Total_neg_score'], self.train['min_score'], self.train['last_nscore'] = features.features_unigrams(self, self.train['tweet_tokens'])
        
        self.train['bigram_positive_score_count'], self.train['bigram_Total_pos_score'], self.train['bigram_max_score'], self.train['bigram_last_pscore'], self.train['bigram_negative_score_count'], self.train['bigram_Total_neg_score'], self.train['bigram_min_score'], self.train['bigram_last_nscore'] = features.features_bigrams(self, self.train['bigrams'])
        
        self.train['upper_case'] = self.train['tweet_tokens'].apply(lambda x: features.no_upper_case(self, x))
        self.train['hashtag_freq'] = self.train['tweet_tokens'].apply(lambda x: features.hashtag_counter(self, x))
                
        train_features = np.array([self.train.positive_score_count, self.train.Total_pos_score, self.train.max_score, self.train.last_pscore, self.train.negative_score_count, self.train.Total_neg_score, self.train.min_score, self.train.last_nscore, self.train.bigram_positive_score_count, self.train.bigram_Total_pos_score, self.train.bigram_max_score, self.train.bigram_last_pscore, self.train.bigram_negative_score_count, self.train.bigram_Total_neg_score, self.train.bigram_min_score, self.train.bigram_last_nscore, self.train.upper_case, self.train.hashtag_freq])

        train_features = train_features.transpose()
        train_features = sparse.coo_matrix(train_features)
        X = hstack((X, train_features.astype(float)))  #stacking ngram with other features (Lex+Enc)
        X_test = cv.transform(self.test['tweet_tokens'])
        Y_test = np.array(self.test['label'].values.tolist())
        
        #creating columns for lexicon based features for test data
        self.test['positive_score_count'], self.test['Total_pos_score'], self.test['max_score'], self.test['last_pscore'], self.test['negative_score_count'], self.test['Total_neg_score'], self.test['min_score'], self.test['last_nscore'] = features.features_unigrams(self, self.test['tweet_tokens'])
        
        self.test['bigram_positive_score_count'], self.test['bigram_Total_pos_score'], self.test['bigram_max_score'], self.test['bigram_last_pscore'], self.test['bigram_negative_score_count'], self.test['bigram_Total_neg_score'], self.test['bigram_min_score'], self.test['bigram_last_nscore'] = features.features_bigrams(self, self.test['bigrams'])
        
        self.test['upper_case'] = self.test['tweet_tokens'].apply(lambda x: features.no_upper_case(self, x))
        self.test['hashtag_freq'] = self.test['tweet_tokens'].apply(lambda x: features.hashtag_counter(self, x))
                
        test_features = np.array([self.test.positive_score_count, self.test.Total_pos_score, self.test.max_score, self.test.last_pscore, self.test.negative_score_count, self.test.Total_neg_score, self.test.min_score, self.test.last_nscore, self.test.bigram_positive_score_count, self.test.bigram_Total_pos_score, self.test.bigram_max_score, self.test.bigram_last_pscore, self.test.bigram_negative_score_count, self.test.bigram_Total_neg_score, self.test.bigram_min_score, self.test.bigram_last_nscore, self.test.upper_case, self.test.hashtag_freq])

        test_features = test_features.transpose()
        test_features = sparse.coo_matrix(test_features)
        X_test = hstack((X_test, test_features.astype(float)))  #stacking ngram with other features (Lex+Enc)
        return(X, Y, X_test, Y_test)
    
    #Model4:Function that will return X_train, Y_train, X_test, Y_test if model is given to be Custom Model (TF-IDF)
    def Custom(self):
        self.train.loc[self.train['label'] == "objective", 'label'] = "neutral"
        self.test.loc[self.test['label'] == "objective", 'label'] = "neutral"
        tweet_corpus = self.train['tweet_tokens']
        tweet_corpus = features.lemmatize(self, tweet_corpus)
        X = np.array(tweet_corpus)
        Y = np.array(self.train['label'].values.tolist())
        token = RegexpTokenizer(r'[a-zA-Z0-9]+')
        tv = TfidfVectorizer(ngram_range = (1,4), sublinear_tf = True, max_features = 40000, tokenizer = token.tokenize)#Implementing TF-IDF 
        X = tv.fit_transform(X)
        
        #obtaining bigrams from tweets   
        train_bigram_list = []
        test_bigram_list = []
        vectorizer = CountVectorizer(ngram_range = (2,2))
        for i in range(len(self.train["tweet_tokens"])):
            vectorizer.fit([self.train.iloc[i]["tweet_tokens"]])
            train_bigram_list.append(vectorizer.get_feature_names())
        for i in range(len(self.test["tweet_tokens"])):
            vectorizer.fit([self.test.iloc[i]["tweet_tokens"]])
            test_bigram_list.append(vectorizer.get_feature_names())
            
        self.train['bigrams'] = train_bigram_list
        self.test['bigrams'] = test_bigram_list
        
        #creating columns for lexicon based features for training data
        self.train['positive_score_count'], self.train['Total_pos_score'], self.train['max_score'], self.train['last_pscore'], self.train['negative_score_count'], self.train['Total_neg_score'], self.train['min_score'], self.train['last_nscore'] = features.features_unigrams(self, self.train['tweet_tokens'])
        
        self.train['bigram_positive_score_count'], self.train['bigram_Total_pos_score'], self.train['bigram_max_score'], self.train['bigram_last_pscore'], self.train['bigram_negative_score_count'], self.train['bigram_Total_neg_score'], self.train['bigram_min_score'], self.train['bigram_last_nscore'] = features.features_bigrams(self, self.train['bigrams'])
        
        self.train['upper_case'] = self.train['tweet_tokens'].apply(lambda x: features.no_upper_case(self, x))
        self.train['hashtag_freq'] = self.train['tweet_tokens'].apply(lambda x: features.hashtag_counter(self, x))
        
        train_features = np.array([self.train.positive_score_count, self.train.Total_pos_score, self.train.max_score, self.train.last_pscore, self.train.negative_score_count, self.train.Total_neg_score, self.train.min_score, self.train.last_nscore, self.train.bigram_positive_score_count, self.train.bigram_Total_pos_score, self.train.bigram_max_score, self.train.bigram_last_pscore, self.train.bigram_negative_score_count, self.train.bigram_Total_neg_score, self.train.bigram_min_score, self.train.bigram_last_nscore, self.train.upper_case, self.train.hashtag_freq])

        train_features = train_features.transpose()
        train_features = sparse.coo_matrix(train_features)
        skb = SelectKBest(chi2, k=500)#selecting the K best features
        X = hstack((X, train_features.astype(float)))      #stacking ngram with other features (Lex+Enc)
        X = skb.fit_transform(X, Y)
        
        X_test = tv.transform(self.test['tweet_tokens'])
        Y_test = np.array(self.test['label'].values.tolist())
        
        #creating columns for lexicon based features for test data
        self.test['positive_score_count'], self.test['Total_pos_score'], self.test['max_score'], self.test['last_pscore'], self.test['negative_score_count'], self.test['Total_neg_score'], self.test['min_score'], self.test['last_nscore'] = features.features_unigrams(self, self.test['tweet_tokens'])
        
        self.test['bigram_positive_score_count'], self.test['bigram_Total_pos_score'], self.test['bigram_max_score'], self.test['bigram_last_pscore'], self.test['bigram_negative_score_count'], self.test['bigram_Total_neg_score'], self.test['bigram_min_score'], self.test['bigram_last_nscore'] = features.features_bigrams(self, self.test['bigrams'])
        
        self.test['upper_case'] = self.test['tweet_tokens'].apply(lambda x: features.no_upper_case(self, x))
        self.test['hashtag_freq'] = self.test['tweet_tokens'].apply(lambda x: features.hashtag_counter(self, x))
        
        test_features = np.array([self.test.positive_score_count, self.test.Total_pos_score, self.test.max_score, self.test.last_pscore, self.test.negative_score_count, self.test.Total_neg_score, self.test.min_score, self.test.last_nscore, self.test.bigram_positive_score_count, self.test.bigram_Total_pos_score, self.test.bigram_max_score, self.test.bigram_last_pscore, self.test.bigram_negative_score_count, self.test.bigram_Total_neg_score, self.test.bigram_min_score, self.test.bigram_last_nscore, self.test.upper_case, self.test.hashtag_freq])

        test_features = test_features.transpose()
        test_features = sparse.coo_matrix(test_features)
        X_test = hstack((X_test, test_features.astype(float))) #stacking ngram with other features (Lex+Enc)
        X_test = skb.transform(X_test)
        return(X, Y, X_test, Y_test)
