from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example_sen="This is an example. vshowing off stop word filtration.kunwar?"
stop_words=set(stopwords.words("english"))
words=word_tokenize(example_sen)
filtered_sen=[w for w in words if   not w in stop_words]
print(filtered_sen)
