import re
#import nltk
import spacy
nlp=spacy.load("en_core_web_sm")
#nltk.download('stopwords')
#nltk.download('stopwords')
from nltk.corpus import stopwords

from collections import Counter

from nltk import ngrams
#import preprocessor as p
import pandas as pd
import networkx as nx



from pke.unsupervised.graph_based.textrank import TextRank


class SingleRank(TextRank):
    """SingleRank keyphrase extraction model.
    This model is an extension of the TextRank model that uses the number of
    co-occurrences to weigh edges in the graph.
    Parameterized example::
        import pke
        # define the set of valid Part-of-Speeches
        pos = {'NOUN', 'PROPN', 'ADJ'}
        # 1. create a SingleRank extractor.
        extractor = pke.unsupervised.SingleRank()
        # 2. load the content of the document.
        extractor.load_document(input='path/to/input',
                                language='en',
                                normalization=None)
        # 3. select the longest sequences of nouns and adjectives as candidates.
        extractor.candidate_selection(pos=pos)
        # 4. weight the candidates using the sum of their word's scores that are
        #    computed using random walk. In the graph, nodes are words of
        #    certain part-of-speech (nouns and adjectives) that are connected if
        #    they occur in a window of 10 words.
        extractor.candidate_weighting(window=10,
                                      pos=pos)
        # 5. get the 10-highest scored candidates as keyphrases
        keyphrases = extractor.get_n_best(n=10)
    """

    def __init__(self):
        """Redefining initializer for SingleRank."""

        super(SingleRank, self).__init__()
        self.graph = nx.Graph()

    def is_redundant(self, candidate, prev, minimum_length=1):
        """Test if one candidate is redundant with respect to a list of already
        selected candidates. A candidate is considered redundant if it is
        included in another candidate that is ranked higher in the list.
        Args:
            candidate (str): the lexical form of the candidate.
            prev (list): the list of already selected candidates (lexical
                forms).
            minimum_length (int): minimum length (in words) of the candidate
                to be considered, defaults to 1.
        """

        # get the tokenized lexical form from the candidate
        candidate = self.candidates[candidate].lexical_form

        # only consider candidate greater than one word
        if len(candidate) < minimum_length:
            return False

        # get the tokenized lexical forms from the selected candidates
        prev = [self.candidates[u].lexical_form for u in prev]
        prev1=[]
        for each_list in prev:
            prev1.append(list(set(each_list)))

        # loop through the already selected candidates
        for prev_candidate in prev1:
            for i in range(len(prev_candidate) - len(candidate) + 1):
                if candidate == prev_candidate[i:i + len(candidate)]:
                    return True
        return False


    def remov_duplicates(self,input):

        # split input string separated by space
        input = input.split(" ")

        # joins two adjacent elements in iterable way
        for i in range(0, len(input)):
            input[i] = "".join(input[i])

        # now create dictionary using counter method
        # which will have strings as key and their
        # frequencies as value
        UniqW = Counter(input)

        # joins two adjacent elements in iterable way
        s = " ".join(UniqW.keys())
        return(s)

    def get_n_best(self, n=10, redundancy_removal=False, stemming=False):
        """Returns the n-best candidates given the weights.
        Args:
            n (int): the number of candidates, defaults to 10.
            redundancy_removal (bool): whether redundant keyphrases are
                filtered out from the n-best list, defaults to False.
            stemming (bool): whether to extract stems or surface forms
                (lowercased, first occurring form of candidate), default to
                False.
        """

        # sort candidates by descending weight
        best = sorted(self.weights, key=self.weights.get, reverse=True)

        # remove redundant candidates
        if redundancy_removal:

            # initialize a new container for non redundant candidates

            non_redundant_best = []

            # loop through the best candidates
            for candidate in best:

                # test wether candidate is redundant
                if self.is_redundant(candidate, non_redundant_best):
                    continue

                # add the candidate otherwise
                non_redundant_best.append(candidate)

                # break computation if the n-best are found
                if len(non_redundant_best) >= n:
                    break

            # copy non redundant candidates in best container
            best = non_redundant_best

        # get the list of best candidates as (lexical form, weight) tuples
        n_best = [(u, self.weights[u]) for u in best[:min(n, len(best))]]

        # replace with surface forms if no stemming
        if not stemming:
            n_best = [(' '.join(self.candidates[u].surface_forms[0]).lower(),
                       self.weights[u]) for u in best[:min(n, len(best))]]

        # return the list of best candidates

        n_best1=[]
        for (each_word,value) in n_best:
            s=self.remov_duplicates(each_word)
            n_best1.append((s,value))

        return n_best1
    def build_word_graph(self, window=10, pos=None):
        """Build a graph representation of the document in which nodes/vertices
        are words and edges represent co-occurrence relation. Syntactic filters
        can be applied to select only words of certain Part-of-Speech.
        Co-occurrence relations can be controlled using the distance (window)
        between word occurrences in the document.
        The number of times two words co-occur in a window is encoded as *edge
        weights*. Sentence boundaries **are not** taken into account in the
        window.
        Args:
            window (int): the window for connecting two words in the graph,
                defaults to 10.
            pos (set): the set of valid pos for words to be considered as nodes
                in the graph, defaults to ('NOUN', 'PROPN', 'ADJ').
        """

        if pos is None:
            pos = {'NOUN', 'PROPN', 'ADJ'}

        # flatten document as a sequence of (word, pass_syntactic_filter) tuples

        text = [(word, sentence.pos[i] in pos) for sentence in self.sentences
                for i, word in enumerate(sentence.stems)]
        #print(type(text))
        # add nodes to the graph
        word2=[]
        word1=[]
        text2=[]
        for (word,valid) in text:

          if word!='\n' and valid:
            word1.append(word)
          if word=='\n':
            word2.append(word1)
            word1=[]
            continue
        word3=[x for x in word2 if x]

        for eachlist in word3:
            for i, word in enumerate(eachlist):
                if word=='rt':
                    continue
                if i<len(eachlist)-1:
                   word2=eachlist[i+1]
                   text2.append((word,word2))
        #print(text2)
        self.graph.add_nodes_from([(word1,word2) for (word1,word2) in text2])

        # add edges to the graph
        # add edges to the graph
        for i, (node1, node2) in enumerate(text2):

            for j in range(i + 1, min(i + window, len(text2))):
                (node3, node4) = text2[j]
                if node2 == node3:
                    if not self.graph.has_edge((node1, node2),(node3, node4)):
                        self.graph.add_edge((node1, node2),(node3, node4),weight=0.0)
                    self.graph[(node1, node2)][(node3, node4)]['weight'] += 1.0




       # for i, (node1, is_in_graph1) in enumerate(text):
#
            # speed up things
 #           if not is_in_graph1:
  #              continue

   #         for j in range(i + 1, min(i + window, len(text))):
    #            node2, is_in_graph2 = text[j]
     #           if is_in_graph2 and node1 != node2:
      #              if not self.graph.has_edge(node1, node2):
        #                self.graph.add_edge(node1, node2, weight=0.0)
         #           self.graph[node1][node2]['weight'] += 1.0

    def candidate_weighting(self, window=10, pos=None, normalized=False):
        """Keyphrase candidate ranking using the weighted variant of the
        TextRank formulae. Candidates are scored by the sum of the scores of
        their words.
        Args:
            window (int): the window within the sentence for connecting two
                words in the graph, defaults to 10.
            pos (set): the set of valid pos for words to be considered as nodes
                in the graph, defaults to ('NOUN', 'PROPN', 'ADJ').
            normalized (False): normalize keyphrase score by their length,
                defaults to False.
        """

        if pos is None:
            pos = {'NOUN', 'PROPN', 'ADJ'}

        # build the word graph
        self.build_word_graph(window=window, pos=pos)

        # compute the word scores using random walk
        w = nx.pagerank_scipy(self.graph,
                              alpha=0.85,
                              tol=0.0001,
                              weight='weight')

        # loop through the candidates
        for k in self.candidates.keys():
            tokens = self.candidates[k].lexical_form
            tokens=(' ').join(tokens)
            tokens_value=[]
            bigram_candidate=ngrams(tokens.split(), 2)
            for each_val in bigram_candidate:
                tokens_value.append(each_val)
            tempval=0
            j=0
            for t in tokens_value:
                try:
                    tempval=tempval+w[t]
                except:
                    tempval=tempval+0
                    j=j+1
            self.weights[k]=tempval
            if normalized:
                self.weights[k] /= len(tokens_value)

            # use position to break ties
            self.weights[k] += (self.candidates[k].offsets[0] * 1e-8)

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
    extractor = SingleRank()

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


        # 2. load the content of the document.
    extractor.load_document(input="tweet.csv",
                                    language='en_core_web_sm',
                                    normalization=None)
    # 3. select the longest sequences of nouns and adjectives as candidates.
    extractor.candidate_selection(pos=pos)

            # 4. weight the candidates using the sum of their word's scores that are
            #    computed using random walk. In the graph, nodes are words of
            #    certain part-of-speech (nouns and adjectives) that are connected if
            #    they occur in a window of 10 words.
    extractor.candidate_weighting(window=2,
                                          pos=pos)

            # 5. get the 10-highest scored candidates as keyphrases
    #f=int(input("Enter how many keyphrases you want to extract="))
    keyphrases = extractor.get_n_best(n)
    keyout.append(keyphrases)


keyphrasesdata=pd.DataFrame()
keyphrasesdata['File_Name']=listoffile
keyphrasesdata['output']=keyout

fileinput=str(input("Enter The Folder Name where you want to save the file for ranking result="))
keyphrasesdata.to_csv('Data/'+ fileinput +'/keyphrases_Singlerank_biword.csv',index=False)

