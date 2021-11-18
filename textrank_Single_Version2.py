import re
#import nltk
import spacy
import pke
nlp=spacy.load("en_core_web_sm")
#nltk.download('stopwords')
from nltk.corpus import stopwords
import pandas as pd
keyout=[]
import glob
tweet_file=str(input("Please enter the folder name from where the tweet file should be taken="))
listoffile=(glob.glob("Data/"+ tweet_file+ "/*.txt"))
n=int(input("Enter how many keyphrases you want to extract="))
for eachfilename in listoffile:
    valueofk=10
    dusertweet=dict()
    usertweet=dict()
    i=0

    with open(eachfilename, "r") as f:
        usertweet=eval(f.read())
    ##with open("newtopic.txt", "r") as f:
    ##    usertweet=eval(f.read())



    j=1
    for each in usertweet.values():
        dusertweet[j]=repr(each)
        j=j+1

    from nltk.tokenize import WordPunctTokenizer
    from bs4 import BeautifulSoup
    tok = WordPunctTokenizer()
    pat1 = r'@[A-Za-z0-9]+'
    pat2 = r'https?://[A-Za-z0-9./]+'
    combined_pat = r'|'.join((pat1, pat2))
    def tweet_cleaner(text):
        soup = BeautifulSoup(text, 'lxml')
        souped = soup.get_text()
        stripped = re.sub(combined_pat, '', souped)
        try:
            clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")
        except:
            clean = stripped
        letters_only = re.sub("[^a-zA-Z]", " ", clean)
        lower_case = letters_only.lower()
        # During the letters_only process two lines above, it has created unnecessay white spaces,
        # I will tokenize and join together to remove unneccessary white spaces
        words = tok.tokenize(lower_case)
        return (" ".join(words)).strip()

    for k,v in dusertweet.items():
        tempstorage=(dusertweet[k])
        tempstorage=tweet_cleaner(tempstorage)
        tempstorage=''.join([i for i in tempstorage if not i.isdigit()])
        tempstorage = re.sub(r"http\S+", "", tempstorage)
        tempstorage=(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tempstorage))
        for w in tempstorage:
             if (w not in stopwords.words("english")) and (len(w)>2):
                w=w.lower()
             else:
                w=""

        dusertweet[k]=tempstorage

    result=[]
    pos = {'NOUN', 'PROPN', 'ADJ'}
    for k, each in dusertweet.items():
        result.append(each)

    tweet=pd.DataFrame()
    tweet['Tweet_data']=result
    tweet.to_csv('tweet.csv',index=False)
    #lstTweet=list(tweet['Tweet_data'])


    #from nltk.corpus import stopwords
    # define the set of valid Part-of-Speeches
    pos = {'NOUN', 'PROPN', 'ADJ'}

    # 1. create a TextRank extractor.
    extractor = pke.unsupervised.TextRank()

    # 2. load the content of the document.
    extractor.load_document(input='tweet.csv',
                            language='en_core_web_sm',
                            normalization=None)

    #3. build the graph representation of the document and rank the words.
    #    Keyphrase candidates are composed from the 33-percent
    #    highest-ranked words.

    extractor.candidate_weighting(window=2,
                                  pos=pos,
                                  top_percent=0.33)
    # 4. get the 10-highest scored candidates as keyphrases
    #f=int(input("Enter how many keyphrases you want to extract="))
    keyphrases = extractor.get_n_best(n)
    keyout.append(keyphrases)

keyphrasesdata=pd.DataFrame()
keyphrasesdata['File_Name']=listoffile
keyphrasesdata['output']=keyout

fileinput=str(input("Enter The Folder Name where you want to save the file for ranking result="))
keyphrasesdata.to_csv('Data/'+ fileinput +'/Textrank_Single.csv',index=False)


