from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps=PorterStemmer()

example_words=["python","pythonre","pythoned","making","make"]
for s in example_words:
    print(ps.stem(s))
