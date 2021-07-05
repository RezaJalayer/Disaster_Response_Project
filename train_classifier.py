import sys
import nltk
nltk.download(['punkt', 'wordnet','stopwords'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


from sqlalchemy import create_engine
import pickle


def load_data(database_filepath):
    
    """ 
    loads the data from DisasterData table in database, and extracts X and Y for training the model
    
    inputs: database_filepath, path to the data base containing the DisasterData table
    outputs: X, data frame of input message for training 
             Y, data frame of category labels for training
             category_names, list of categories
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterData',con=engine)
    X = df["message"]
    Y = df.drop(columns=['id','message','original','genre'])
    category_names = Y.columns.tolist()
    return X,Y, category_names


def tokenize(text):
    
    """
    Tokenize the input text, removing stop words and lemmatizing
    
    input: text, the text string to be tokenized
    output: listof cleaned words(tokens)
    """
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words("english")

    clean_tokens = []
    for tok in tokens:
        if tok not in stop_words:
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    
    """
    build a machine learning pipeline: CountVectorize-->Tfidf-->estimator
    Grid serach is done to find best parameters
    
    output: model with best parameters
    """
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier())
    ])
    parameters = {
              'vect__max_features':(None,100,1000),
              'clf__max_depth': (None,5),
              }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    
    """
    printing the performance metrics for each gategory 
    inputs: model, estimator artifacts
            X_test, test data
            Y_test, test labels
            category_names, list of categories
    """
            
    y_pred = model.predict(X_test)

    j = 0
    for col in category_names:
        print(classification_report(Y_test[:][col].values, y_pred[:,j],target_names=[col+ '0',col+ '1']))
        j=j+1

def save_model(model, model_filepath):
    
    """
    save model to disk
    inputs: model, the atifacts to be saved
            model_filepath, path and name of the saved file
    """
    
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()