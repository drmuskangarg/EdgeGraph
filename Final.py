import pandas as pd
import glob
import re
import csv


import nltk
import Original_nerank_biword_Version2
#import original_nerank_single_word_version2
import positional_rank_Single_Vesion2
import Positionalrank_biword_Version2
import proposed_nerank_biword_Version2
import proposed_nerank_single_word_version2
import Singlerank_biword_Version2
import singlerank_single_vesion2
import textrank_biword_version2
import textrank_Single_Version2
import singletpr_biword_version2
import singletpr_singleword_Version2
import topicrank_biword_version3
import topicrank_single_Version2
#from textrank_Single_Version2 import filename
def printSubsInDelimeters(str):

    # Regex to extract the
    # between two delimiters
    regex = "\\((.*?)\\)"

    # Find match between given string
    # and regular expression
    # using re.findall()
    matches = re.findall(regex, str)
    return(matches)
    # Print the matches
    #for match in matches:
     #   print(match)
a=[]
b=[]
e=[]
n=filename4=int(input("Please Enter The number of Keyphrases Extracted="))
filename4=str(input("Please Enter The folder Name where all the Tweet files are stored="))
listoffile2=(glob.glob("Data/" + filename4 + "/*.txt"))
filename=str(input("Please Enter The folder Name where all the methods result files are stored="))
listoffile1=(glob.glob("Data/" + filename + "/*.csv"))

for eachfilename1 in listoffile1:
    filename=re.search('/outputthesis/(.+?).csv', eachfilename1)
    a.append(filename.group(1))


for eachfilename1 in listoffile1:
    b.append(pd.read_csv(eachfilename1))

for eachfilename1 in listoffile2:
    filename=re.search('/100Thesis/(.+?).txt', eachfilename1)
    e.append(filename.group(1))
keyphrases=["Method_Name"]
s=["Keyphrase%d" % i for i in range(1,n+1)]
for each in s:
    keyphrases.append(each)
#df1=pd.DataFrame(index=(a),columns=(keyphrases))
for j in range(0,len(e)):
    df=pd.DataFrame(index=(a),columns=(keyphrases))

    for i in range(0,14):
        df.xs(a[i])['Method_Name']=a[i]
        c=b[i]
        str1=c.iloc[j]['output']
        str2=(re.findall(re.escape('[')+"(.*)"+re.escape(']'),str1)[0])
        d=printSubsInDelimeters(str2)
        for k in range(0,len(d)):
            df.xs(a[i])[keyphrases[k+1]]=d[k]


    df.to_csv('Data/Keyphrases_Required1/finaloutput_100thesis.csv',mode='a',index=False)

with open("Data/Keyphrases_Required1/finaloutput_100thesis.csv", 'r') as infile:
    read=csv.reader(infile, delimiter=',')
    with open("Data/Keyphrases_Required1/finaloutput_100thesis1.csv", 'wt') as output:
        outwriter=csv.writer(output, delimiter=',')
        i=0
        for row in read:


            if 'Method_Name' in row:
                outwriter.writerow([])
                outwriter.writerow([])
                row1=e[i]
                outwriter.writerow([row1])
                outwriter.writerow(row)
                i+=1
            else:

                outwriter.writerow(row)





