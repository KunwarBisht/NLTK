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

short_pos=open("positive.txt" ,"r").read()
short_neg=open("negative.txt" ,"r").read()
documents=[]
all_words=[]
#  j is adject, r is adverb, and v is verb
#allowed_word_types = ["J","R","V"]
allowed_word_types = ["J"]

for p in short_pos.split('\n'):
    documents.append( (p, "pos") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

    
for p in short_neg.split('\n'):
    documents.append( (p, "neg") )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())


#documents pickle
doc_pickle=open("document.pickle", "wb")
pickle.dump(documents,doc_pickle)
doc_pickle.close()

all_words=nltk.FreqDist(all_words)

word_features=list(all_words.keys())[:5000]


word_features_pickle=open("word_features.pickle","wb")
pickle.dump(word_features,word_features_pickle)
word_features_pickle.close()


def find_features(document):
    words=word_tokenize(document)
    features={}
    for w in word_features:
        features[w]= (w in words)
         
    return features
#print((find_features(movie_reviews.words("neg/cv000_29416.txt"))))

featuresets = [(find_features(rev), category) for (rev, category) in documents]  #review each word is review and category

featuresets_save=open("featuresets.pickle","wb")
pickle.dump(featuresets ,featuresets_save)
featuresets_save.close()

random.shuffle(featuresets)

# data example:      
training_set = featuresets[:10000]
testing_set =  featuresets[10000:]


classifier=nltk.NaiveBayesClassifier.train(training_set)
#call saved classifier for accuracy
##classifier_f=open("naivebayes.pickle","rb")
##classifier=pickle.load(classifier_f)
##classifier_f.close()


###saving classifer data in pickle
save_classifier=open("naivebayes.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

#scikit_learn with nltk chapeter 15

##MultinomialNB
MNB_classifier =SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MultinomialNB accuracy %",(nltk.classify.accuracy(MNB_classifier,testing_set))*100)

save_classifier=open("MNB_classifier.pickle","wb")
pickle.dump(MNB_classifier, save_classifier)
save_classifier.close()
#####

##BernoulliNB_classifier
BernoulliNB_classifier =SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy %",(nltk.classify.accuracy(BernoulliNB_classifier,testing_set))*100)

save_classifier=open("BernoulliNB_classifier.pickle","wb")
pickle.dump(BernoulliNB_classifier, save_classifier)
save_classifier.close()

#####

##LogisticRegression
LogisticRegression_classifier =SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression accuracy %",(nltk.classify.accuracy(LogisticRegression_classifier,testing_set))*100)
save_classifier=open("LogisticRegression_classifier.pickle","wb")
pickle.dump(LogisticRegression_classifier, save_classifier)
save_classifier.close()

#####


##SGDClassifier
SGDClassifier_classifier =SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier accuracy %",(nltk.classify.accuracy(SGDClassifier_classifier,testing_set))*100)
save_classifier=open("SGDClassifier_classifier.pickle","wb")
pickle.dump(SGDClassifier_classifier, save_classifier)
save_classifier.close()
#####

##SVC
SVC_classifier =SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC accuracy %",(nltk.classify.accuracy(SVC_classifier,testing_set))*100)
save_classifier=open("SVC_classifier.pickle","wb")
pickle.dump(SVC_classifier, save_classifier)
save_classifier.close()
#####

##LinearSVC
LinearSVC_classifier =SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC accuracy %",(nltk.classify.accuracy(LinearSVC_classifier,testing_set))*100)

save_classifier=open("LinearSVC_classifier.pickle","wb")
pickle.dump(LinearSVC_classifier, save_classifier)
save_classifier.close()
#####

##NuSVC
NuSVC_classifier =SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC accuracy %",(nltk.classify.accuracy(NuSVC_classifier,testing_set))*100)

save_classifier=open("NuSVC_classifier.pickle","wb")
pickle.dump(NuSVC_classifier, save_classifier)
save_classifier.close()
#####

voted_classifier = VoteClassifier(classifier,
                                  NuSVC_classifier,
                                  LinearSVC_classifier,
                                  SGDClassifier_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)

print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)













