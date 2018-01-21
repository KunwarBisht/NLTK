from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
from short_reviews18 import sentiment_mod as s
ckey='sZOUBTRXI3qeSvYCfhENrnPhQ'
csecret='q6ExMtZaAdNJpLeqyHDKJkdrqPoGaDMMRHFrTKCCbx3SeCvWO3'
atoken='2469666950-MXFBTdrIytafLj5BUaFKKCRhKfcZWTCIi0fVodw'
asecret='Y0wv9tOsJYJK2Qcww4vT58winpwF8C9BngwGq4IfAyWdp'

class listener(StreamListener):

    def on_data(self, data):
        all_data=json.loads(data)
        tweet =all_data["text"]
        sentiment_value ,confidence =s.sentiment(tweet)
        print(tweet , sentiment_value ,confidence)

        if confidence *100 >=80:
            output =open("twiter-out.txt","a")
            output.write(sentiment_value)
            output.write('\n')
            output.close()

        return(True)

    def on_error(self, status):
        print (status)

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["tensorflow"])
