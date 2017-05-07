import numpy as np
import pandas as pd
import re
from gensim import corpora
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import SnowballStemmer


if __name__ == "__main__":

    #load data
    train_df = pd.read_csv('train.tsv', sep='\t', header=0)
    test_df = pd.read_csv('test.tsv', sep='\t', header=0)

    SentenceId=train_df['SentenceId'].values
    raw_docs_train=train_df['Phrase'].values
    sentiment_train=train_df['Sentiment'].values

    #text pre-processing
    stop_words = set(stopwords.words('english'))
    stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])
    stemmer = SnowballStemmer('english')

    print "pre-processing train docs..."
    processed_docs_train = []
    for doc in raw_docs_train:
        #doc=re.sub("[^a-zA-Z]"," ",doc)
        doc=doc.lower()
        tokens = word_tokenize(doc)
        stemmed=tokens
        #stemmed = [stemmer.stem(word) for word in tokens]
        processed_docs_train.append(stemmed)
   
    Clean_train_doc=np.asarray(processed_docs_train)
    sentiment_train=np.asarray(sentiment_train)

    np.save('Clean_train_doc',Clean_train_doc)
    np.save('sentiment_train',sentiment_train)
