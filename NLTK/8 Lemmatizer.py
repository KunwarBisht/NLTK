from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()

print(lemmatizer.lemmatize("Persons", pos='n'))
print(lemmatizer.lemmatize("goodesy"))
print(lemmatizer.lemmatize("hello"))
