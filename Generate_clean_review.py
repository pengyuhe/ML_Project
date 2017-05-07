import numpy as np
import pandas as pd
import re
from gensim import corpora
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import SnowballStemmer


def Clean_doc(raw_docs,stemmer,stop_words):
    
    processed_docs = []
    for doc in raw_docs:
        #doc=re.sub("[^a-zA-Z]"," ",doc)
        doc=doc.lower()
        tokens = word_tokenize(doc)
        stemmed=tokens
        #stemmed = [stemmer.stem(word) for word in tokens]
        processed_docs.append(stemmed)

    return np.asarray(processed_docs)

if __name__ == "__main__":

    #load data
    train_df = pd.read_csv('train.tsv', sep='\t', header=0)
    test_df = pd.read_csv('test.tsv', sep='\t', header=0)

    SentenceId=train_df['SentenceId'].values
    raw_docs_train=train_df['Phrase'].values
    sentiment_train=train_df['Sentiment'].values
    raw_docs_test=test_df['Phrase'].values

    #text pre-processing
    stop_words = set(stopwords.words('english'))
    stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])
    stemmer = SnowballStemmer('english')

    print "pre-processing train docs..."
  
    Clean_train_doc= Clean_doc(raw_docs_train,stemmer,stop_words)
    Clean_test_doc = Clean_doc(raw_docs_test,stemmer,stop_words)
    sentiment_train=np.asarray(sentiment_train)

    np.save('Clean_train_doc',Clean_train_doc)
    np.save('sentiment_train',sentiment_train)
    np.save('Clean_test_doc',Clean_test_doc)
