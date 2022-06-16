import argparse
import features
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', dest='train', required=True,
                        help='Full path to the training file')
    parser.add_argument('--test', dest='test', required=True,
                        help='Full path to the evaluation file')
    parser.add_argument('--model', dest='model', required=True,
                        choices=["Ngram", "Ngram+Lex", "Ngram+Lex+Enc", "Custom"],
                        help='The name of the model to train and evaluate.')
    parser.add_argument('--lexicon_path', dest='lexicon_path', required=True,
                        help='The full path to the directory containing the lexica.'
                             ' The last folder of this path should be "lexica".')
    args = parser.parse_args()

    #Creating an instance of the class features from features.py
    feature = features.features(args.train, args.test, args.model, args.lexicon_path)
    
    #If model is Ngram
    if args.model == "Ngram":
        X_train, Y_train, X_test, Y_test = feature.Ngram() # obtaining train and test data from features.py
        #Best performing classifier - SVM with a C value of 10
        svm = LinearSVC(C=10)
        svm.fit(X_train, Y_train) #Fitting train data and labels into model
        print(f1_score(Y_test, svm.predict(X_test), average="macro"))#Printing Macro F1 score
        print(f1_score(Y_test, svm.predict(X_test), average= None))#Printing class-wise F1 score
        
    #If model is Ngram+Lex
    if args.model == "Ngram+Lex":
        X_train, Y_train, X_test, Y_test = feature.Ngram_Lex()# obtaining train and test data from features.py
        #Best performing classifier - SVM with a C value of 10
        svm = LinearSVC(C=10)
        svm.fit(X_train, Y_train) #Fitting train data and labels into model
        print(f1_score(Y_test, svm.predict(X_test), average="macro"))#Printing Macro F1 score
        print(f1_score(Y_test, svm.predict(X_test), average= None))#Printing class-wise F1 score

    #If model is Ngram+Lex+Enc    
    if args.model == "Ngram+Lex+Enc":
        X_train, Y_train, X_test, Y_test = feature.Ngram_Lex_Enc()# obtaining train and test data from features.py
        #Best performing classifier - SVM with a C value of 10
        
        svm = LinearSVC(C=10)
        svm.fit(X_train, Y_train)#Fitting train data and labels into model
        print(f1_score(Y_test, svm.predict(X_test), average="macro"))#Printing Macro F1 score
        print(f1_score(Y_test, svm.predict(X_test), average= None))#Printing class-wise F1 score
#       
    #If model is Custom    
    if args.model == "Custom":
        X_train, Y_train, X_test, Y_test = feature.Custom()# obtaining train and test data from features.py
        #Best performing classifier - SVM with a C value of 10
        
        svm = LinearSVC(C=10)
        svm.fit(X_train, Y_train)#Fitting train data and labels into model
        print(f1_score(Y_test, svm.predict(X_test), average="macro"))#Printing Macro F1 score
        print(f1_score(Y_test, svm.predict(X_test), average= None))#Printing class-wise F1 score
        

    
    
