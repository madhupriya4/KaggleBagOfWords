import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from KaggleWord2VecUtility import  KaggleWord2VecUtility
import pandas as pd
import numpy as np

if __name__ == '__main__':
    train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0, delimiter="\t", quoting=3)
    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t", quoting=3)

    clean_train_reviews=[]
    for i in xrange(len(train["review"])):
        clean_train_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(train["review"][i], True)))


    vectorizer =CountVectorizer(analyzer="word", tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)
    train_data_features=vectorizer.fit_transform(clean_train_reviews)

    np.asarray(train_data_features)

    print 'training..'

    forest = RandomForestClassifier(n_estimators=100)

    forest=forest.fit(train_data_features,train["sentiment"])

    clean_test_reviews=[]

    print 'parsing test reviews..'

    for i in xrange(0, len(test["review"])):
        clean_test_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(test["review"][i], True)))

    test_data_features=vectorizer.transform(clean_test_reviews)
    np.asarray(test_data_features)

    print 'predicting test labels..'
    result=forest.predict(test_data_features)

    output=pd.DataFrame(data={"id":test["id"],"sentiment":result})

    output.to_csv(os.path.join(os.path.dirname(__file__),'data','Bag_of_Words_model.csv'),index=False,quoting=3)
    print 'done'