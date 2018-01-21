#text classifier
#sentiment analysis algorithm

import nltk
import pickle
import random
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression ,SGDClassifier
from sklearn.svm import SVC,LinearSVC , NuSVC
from nltk.corpus import movie_reviews
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize

#building class

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers =classifiers
       

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
    
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


#documents pickle
doc_pickle=open("document.pickle", "rb")
documents=pickle.load(doc_pickle)
doc_pickle.close()

#word_features 5k features unique first 5k words 
word_features_pickle=open("word_features.pickle","rb")
word_features=pickle.load(word_features_pickle)
word_features_pickle.close()


def find_features(document):
    words=word_tokenize(document)
    features={}
    for w in word_features:
        features[w]= (w in words)
         
    return features



featuresets_pickle=open("featuresets.pickle", "rb")
featuresets=pickle.load(featuresets_pickle)
featuresets_pickle.close()

random.shuffle(featuresets)
#print(len(featuresets))
# data example:      
training_set = featuresets[:10000]
testing_set =  featuresets[10000:]

#classifier=nltk.NaiveBayesClassifier.train(training_set)
#call saved classifier for accuracy
classifier_f=open("naivebayes.pickle","rb")
classifier=pickle.load(classifier_f)
classifier_f.close()

##print("Classifier accuracy %",(nltk.classify.accuracy(classifier,testing_set))*100)
##classifier.show_most_informative_features(15)



##MultinomialNB
classifier_f=open("MNB_classifier.pickle","rb")
MNB_classifier=pickle.load(classifier_f)
classifier_f.close()


#####

##BernoulliNB_classifier
classifier_f=open("BernoulliNB_classifier.pickle","rb")
BernoulliNB_classifier=pickle.load(classifier_f)
classifier_f.close()


#####

##LogisticRegression
classifier_f=open("LogisticRegression_classifier.pickle","rb")
LogisticRegression_classifier=pickle.load(classifier_f)
classifier_f.close()



#####


##SGDClassifier
classifier_f=open("SGDClassifier_classifier.pickle","rb")
SGDClassifier_classifier=pickle.load(classifier_f)
classifier_f.close()

#####

##SVC
classifier_f=open("SVC_classifier.pickle","rb")
SVC_classifier=pickle.load(classifier_f)
classifier_f.close()

#####

##LinearSVC
classifier_f=open("LinearSVC_classifier.pickle","rb")
LinearSVC_classifier=pickle.load(classifier_f)
classifier_f.close()


##NuSVC
classifier_f=open("NuSVC_classifier.pickle","rb")
NuSVC_classifier=pickle.load(classifier_f)
classifier_f.close()



voted_classifier = VoteClassifier(classifier,
                                  SVC_classifier,
                                  NuSVC_classifier,
                                  LinearSVC_classifier,
                                  SGDClassifier_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)

#print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)
def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats),voted_classifier.confidence(feats)













