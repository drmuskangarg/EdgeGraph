import re
#import nltk
import spacy
nlp=spacy.load("en_core_web_sm")
#nltk.download('stopwords')
#nltk.download('stopwords')
from nltk.corpus import stopwords
from collections import Counter
#from nltk import ngrams
#import preprocessor as p
import pandas as pd
import networkx as nx

import math
import logging
from pke.base import LoadFile
TFIDF_dict={}
weight1={}
sum_cent_dc_evc_bc={}
class TextRank(LoadFile):
    """TextRank for keyword extraction.
    This model builds a graph that represents the text. A graph based ranking
    algorithm is then applied to extract the lexical units (here the words) that
    are most important in the text.
    In this implementation, nodes are words of certain part-of-speech (nouns
    and adjectives) and edges represent co-occurrence relation, controlled by
    the distance between word occurrences (here a window of 2 words). Nodes
    are ranked by the TextRank graph-based ranking algorithm in its unweighted
    variant.
    Parameterized example::
        import pke
        # define the set of valid Part-of-Speeches
        pos = {'NOUN', 'PROPN', 'ADJ'}
        # 1. create a TextRank extractor.
        extractor = pke.unsupervised.TextRank()
        # 2. load the content of the document.
        extractor.load_document(input='path/to/input',
                                language='en',
                                normalization=None)
        # 3. build the graph representation of the document and rank the words.
        #    Keyphrase candidates are composed from the 33-percent
        #    highest-ranked words.
        extractor.candidate_weighting(window=2,
                                      pos=pos,
                                      top_percent=0.33)
        # 4. get the 10-highest scored candidates as keyphrases
        keyphrases = extractor.get_n_best(n=10)
    """

    def __init__(self):
        """Redefining initializer for TextRank."""

        super(TextRank, self).__init__()

        self.graph = nx.Graph()
        """The word graph."""

    def candidate_selection(self, pos=None):
        """Candidate selection using longest sequences of PoS.
        Args:
            pos (set): set of valid POS tags, defaults to ('NOUN', 'PROPN',
                'ADJ').
        """

        if pos is None:
            pos = {'NOUN', 'PROPN', 'ADJ'}

        # select sequence of adjectives and nouns
        self.longest_pos_sequence_selection(valid_pos=pos)


    def build_word_graph(self, window=2, pos=None):
        """Build a graph representation of the document in which nodes/vertices
        are words and edges represent co-occurrence relation. Syntactic filters
        can be applied to select only words of certain Part-of-Speech.
        Co-occurrence relations can be controlled using the distance between
        word occurrences in the document.
        As the original paper does not give precise details on how the word
        graph is constructed, we make the following assumptions from the example
        given in Figure 2: 1) sentence boundaries **are not** taken into account
        and, 2) stopwords and punctuation marks **are** considered as words when
        computing the window.
        Args:
            window (int): the window for connecting two words in the graph,
                defaults to 2.
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
        text2=[]
        word2=[]
        word1=[]
        for (word,valid) in text:

          if word!='\n' and valid:
            word1.append(word)
          else:
            word2.append(word1)
            word1=[]
            continue
        word3=[x for x in word2 if x]
        for eachlist in word3:
            for i, word in enumerate(eachlist):

                if i<len(eachlist):

                   text2.append(word)

        self.graph.add_nodes_from([word for word, valid in text if valid])
        TF_dict={}
        #n=len(text2)
        for word1 in text2:
            if word1 in TF_dict.keys():
                TF_dict[word1]=(TF_dict[word1]+1)
            else:
                TF_dict[word1]=1
        n=len(text2)
        TF_dict1 = {k: v / n for k, v in TF_dict.items()}
        idf_dict={}
        N=0

        #print(word3)
        N=len(word3)

        for word1 in TF_dict.keys():
            count=0.0
            for each in word3:
              if word1 in each:
                  count=count+1.0

            idf_dict[word1]=(math.log((N)/(count)))




        for word1 in TF_dict1.keys():
            TFIDF_dict[word1]=(float)(TF_dict1[word1])*(float)(idf_dict[word1])
        #print(TFIDF_dict.keys())
        for i in range(0,len(text2)-1):
            weight1[text2[i],text2[i+1]]=1
        for i in range(0,len(text2)-1):

            for j in range(0,len(text2)-1):
                if(text2[i]==text2[j] and text2[i+1]==text2[j+1] ):
                    weight1[text2[i],text2[i+1]]+=1


        # add edges to the graph
        for i, node1 in enumerate(text2):

            for j in range(i + 1, min(i + window, len(text2))):
                node3 = text2[j]
                if not self.graph.has_edge(node1,node3) and node1 != node3:
                    self.graph.add_edge(node1,node3, weight=weight1[node1,node3])
                    self.graph[node1][node3]['weight'] += 0.0


    def pagerank(self,alpha=0.85,personalization=None,max_iter=100,tol=1.0e-6,nstart=None,weight="weight",dangling=None):
      #print(list(self))
      if len(self.graph) == 0:
          return {}

      if not self.graph.is_directed():
          D = self.graph.to_directed()
      else:
          D = self.graph

      # Create a copy in (right) stochastic form
      W = nx.stochastic_graph(D, weight=weight)
      N = W.number_of_nodes()

      # Choose fixed starting vector if not given
      if nstart is None:
          x = dict.fromkeys(W, 1.0 / N)
      else:
          # Normalized nstart vector
          s = float(sum(nstart.values()))
          x = {k: v / s for k, v in nstart.items()}
          #print('first')
          #print(x)
      if personalization is None:
          # Assign uniform personalization vector if not given
          p = dict.fromkeys(W, 1.0 / N)
      else:
          s = float(sum(personalization.values()))
          p = {k: v / s for k, v in personalization.items()}

      if dangling is None:
          # Use personalization vector if dangling vector not specified

          dangling_weights = p
      else:
          s = float(sum(dangling.values()))
          dangling_weights = {k: v / s for k, v in dangling.items()}
      #print(dangling_weights)
      dangling_nodes = [n for n in W if W.out_degree(n, weight=weight) == 0.0]
      #print(dangling_nodes)
      # power iteration: make up to max_iter iterations
      for _ in range(max_iter):
          xlast = x
          x = dict.fromkeys(xlast.keys(), 0)
          #print(x)
          sumdangle=0
          for n in dangling_nodes:
              sumdangle=sumdangle+xlast[n]
          danglesum = 0.85 * sumdangle
          for n in x:
              # this matrix multiply looks odd because it is
              # doing a left multiply x^T=xlast^T*W
              D_t = sum(W[n][nbr][weight] * xlast[nbr] for nbr in W[n])
              for nbr in W[n]:
                  #x[nbr] += d * xlast[n] * W[n][nbr][weight]

                  x[nbr] += (
                      0.85 * (W[n][nbr][weight] * xlast[nbr] / D_t) * xlast[n]*TFIDF_dict[nbr])+(1.0 - 0.85) * TFIDF_dict[nbr]
              x[n] += danglesum * dangling_weights[n] + (1.0 - 0.85) * p[n]*TFIDF_dict[n]
          # check convergence, l1 norm
          err = sum([abs(x[n] - xlast[n]) for n in x])
          if err < N * tol:
              return x
      raise nx.PowerIterationFailedConvergence(max_iter)
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
    def candidate_weighting(self,
                            window=2,
                            pos=None,
                            top_percent=None,
                            normalized=False):
        """Tailored candidate ranking method for TextRank. Keyphrase candidates
        are either composed from the T-percent highest-ranked words as in the
        original paper or extracted using the `candidate_selection()` method.
        Candidates are ranked using the sum of their (normalized?) words.
        Args:
            window (int): the window for connecting two words in the graph,
                defaults to 2.
            pos (set): the set of valid pos for words to be considered as nodes
                in the graph, defaults to ('NOUN', 'PROPN', 'ADJ').
            top_percent (float): percentage of top vertices to keep for phrase
                generation.
            normalized (False): normalize keyphrase score by their length,
                defaults to False.
        """

        if pos is None:
            pos = {'NOUN', 'PROPN', 'ADJ'}

        # build the word graph
        self.build_word_graph(window=window, pos=pos)

        # compute the word scores using the unweighted PageRank formulae

        w = self.pagerank(self.graph,nstart=TFIDF_dict,weight='weight')

        # generate the phrases from the T-percent top ranked words
        if top_percent is not None:

            # warn user as this is not the pke way of doing it
            logging.warning("Candidates are generated using {}-top".format(
                            top_percent))

            # computing the number of top keywords
            nb_nodes = self.graph.number_of_nodes()
            to_keep = min(math.floor(nb_nodes * top_percent), nb_nodes)

            # sorting the nodes by decreasing scores
            top_words = sorted(w, key=w.get, reverse=True)

            # creating keyphrases from the T-top words
            self.longest_keyword_sequence_selection(top_words[:int(to_keep)])

        # weight candidates using the sum of their word scores
         #weight candidates using the sum of their word scores
        for k in self.candidates.keys():
            tokens = self.candidates[k].lexical_form
            self.weights[k] = sum([w[t] for t in tokens])
            if normalized:
                self.weights[k] /= len(tokens)

            # use position to break ties
            self.weights[k] += (self.candidates[k].offsets[0]*1e-8)

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
    extractor = TextRank()
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



            # define the set of valid Part-of-Speeches
    pos = {'NOUN', 'PROPN', 'ADJ'}
            # 1. create a TextRank extractor.

            # 2. load the content of the document.
    extractor.load_document(input='tweet.csv',
                                    language='en_core_web_sm',
                                    normalization=None)
            # 3. build the graph representation of the document and rank the words.
            #    Keyphrase candidates are composed from the 33-percent
            #    highest-ranked words.
    extractor.candidate_weighting(window=2,pos=pos,top_percent=0.33)
            # 4. get the 10-highest scored candidates as keyphrases

    keyphrases = extractor.get_n_best(n,redundancy_removal=True)
    keyout.append(keyphrases)
    TFIDF_dict={}
    weight1={}
    sum_cent_dc_evc_bc={}
keyphrasesdata=pd.DataFrame()
keyphrasesdata['File_Name']=listoffile
keyphrasesdata['output']=keyout

fileinput=str(input("Enter The Folder Name where you want to save the file for ranking result="))
keyphrasesdata.to_csv('Data/'+ fileinput +'/keyphrases_original_single.csv',index=False)



