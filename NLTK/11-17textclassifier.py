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

documents= [ (list(movie_reviews.words(fileid)),category)
              for category in movie_reviews.categories()
              for fileid in movie_reviews.fileids(category)]

#print(documents[1])  #documents contain 2000 files 1000 each +vn n -ve all words of a file and category
   
#random.shuffle(documents)

all_words=[]

for w in movie_reviews.words():
    all_words.append(w.lower())

all_words=nltk.FreqDist(all_words)
##print(len(all_words))
##print(all_words.most_common(15))
##print(all_words["stupid"])
word_features=list(all_words.keys())[:3000]



def find_features(document):
    words=set(document)
    features={}
    for w in word_features:
        features[w]= (w in words)
         
    return features
#print((find_features(movie_reviews.words("neg/cv000_29416.txt"))))

featuresets = [(find_features(rev), category) for (rev, category) in documents]  #review each word is review and category

#print(len(featuresets))

#set that we'll train our classifier with 

#positive data sets   
##training_set=featuresets[:1800]
##testing_set =featuresets[1800:]

#Negative data sets   
training_set=featuresets[200:]
testing_set =featuresets[:200]

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




     
#classifier=nltk.NaiveBayesClassifier.train(training_set)
#call saved classifier for accuracy
classifier_f=open("naivebayes.pickle","rb")
classifier=pickle.load(classifier_f)
classifier_f.close()

print("Classifier accuracy %",(nltk.classify.accuracy(classifier,testing_set))*100)
classifier.show_most_informative_features(15)
###saving classifer data in pickle
##save_classifier=open("naivebayes.pickle","wb")
##pickle.dump(classifier, save_classifier)
##save_classifier.close()

#scikit_learn with nltk chapeter 15

##MultinomialNB
MNB_classifier =SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MultinomialNB accuracy %",(nltk.classify.accuracy(MNB_classifier,testing_set))*100)
#####

##BernoulliNB_classifier
BernoulliNB_classifier =SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("MultinomialNB accuracy %",(nltk.classify.accuracy(BernoulliNB_classifier,testing_set))*100)
#####

##LogisticRegression
LogisticRegression_classifier =SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression accuracy %",(nltk.classify.accuracy(LogisticRegression_classifier,testing_set))*100)
#####


##SGDClassifier
SGDClassifier_classifier =SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier accuracy %",(nltk.classify.accuracy(SGDClassifier_classifier,testing_set))*100)
#####

##SVC
SVC_classifier =SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC accuracy %",(nltk.classify.accuracy(SVC_classifier,testing_set))*100)
#####

##LinearSVC
LinearSVC_classifier =SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC accuracy %",(nltk.classify.accuracy(LinearSVC_classifier,testing_set))*100)
#####

##NuSVC
NuSVC_classifier =SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC accuracy %",(nltk.classify.accuracy(NuSVC_classifier,testing_set))*100)
#####

voted_classifier = VoteClassifier(classifier,
                                  NuSVC_classifier,
                                  LinearSVC_classifier,
                                  SGDClassifier_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)
print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)

print("Classification:", voted_classifier.classify(testing_set[0][0]), "Confidence %:",voted_classifier.confidence(testing_set[0][0])*100)
print("Classification:", voted_classifier.classify(testing_set[1][0]), "Confidence %:",voted_classifier.confidence(testing_set[1][0])*100)
print("Classification:", voted_classifier.classify(testing_set[2][0]), "Confidence %:",voted_classifier.confidence(testing_set[2][0])*100)
print("Classification:", voted_classifier.classify(testing_set[3][0]), "Confidence %:",voted_classifier.confidence(testing_set[3][0])*100)
print("Classification:", voted_classifier.classify(testing_set[4][0]), "Confidence %:",voted_classifier.confidence(testing_set[4][0])*100)
print("Classification:", voted_classifier.classify(testing_set[5][0]), "Confidence %:",voted_classifier.confidence(testing_set[5][0])*100)











