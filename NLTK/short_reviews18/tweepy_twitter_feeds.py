from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import sentiment_mod as s

ckey='tWx5OSyfBFHNL4vcn2PacQnnd'
csecret='CJ6zi7tQ2SlPogHo1Pro3QXXOw4KgCUsrMZQRQdD9yj4EWKaVO'
atoken='2469666950-7ofExapBOrU3SpsdRHTVIFGhbhgbhFyM6yINE1J'
asecret='jEJH6oz1lZvvnO2CTfhIAlTUnKubG3whlmM5BQhJe4QPb'


class listener(StreamListener):
    def on_data(self, data):
            
        all_data = json.loads(data)
        tweet = all_data["text"]
        sentiment_value, confidence = s.sentiment(tweet)
        tweet = tweet.encode("utf-8", errors='ignore')
        print(tweet, sentiment_value, confidence)
	      
        if confidence*100 >= 80:
                
            output = open("twitter-out.txt","a")
            output.write(sentiment_value)
            output.write('\n')
            output.close()

        return True

    def on_error(self, status):
        print(status)

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())

try:
    twitterStream.filter(track=["Secret Superstar"])
except ValueError:
    print(ValueError)

