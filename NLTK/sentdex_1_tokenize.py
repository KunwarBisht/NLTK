from nltk.tokenize import word_tokenize , sent_tokenize

txt="hello Mr.K how are your and how you doing today? The wheathe seems good and the musice is loud, and beautiful.Cool day!. is't it"

##print(word_tokenize(txt))
##print(sent_tokenize(txt))

for w in sent_tokenize(txt):
    print(w)
    
