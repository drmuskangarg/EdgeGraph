import pandas as pd
from nltk.corpus import stopwords

import glob
import re
import csv
import matplotlib.pyplot as plt
import numpy as np
#from matplotlib.font_manager import FontProperties
def norm(stringdata):
    a=set()
    for each in stringdata:
        each = each.replace(',', ' ')
        each = each.replace('-', ' ')
        each=each.lower()
        a.add(each)
    stop_words = set(stopwords.words('portuguese'))

    stringdata=[]
    for token in a:
        if token not in stop_words and token!='':
            stringdata.append(token)
    return stringdata
def rouge_1(extracted,actual):

    getdata=set(norm(extracted))
    refdata=set(norm(actual))

    getval=len(getdata.intersection(refdata))
    if len(getdata)>0:
        result=(float)((float)(getval)/len(getdata))
    else:
        result=0
    #print (result)
    return result


def dataframe(filename):
    #df=pd.read_csv('/Users/Files From e.localized/Poornima Docs/Machine Learning/Dr. Muskaan Garg/Reserach Paper_1/Keyphrases_Required/finaloutput6.csv')

    listoffile1=(glob.glob("Data/" + filename + "/*.txt"))
    #df3=pd.DataFrame()
    i=0
    for eachfilename1 in listoffile1:
        filename1=re.search('/Reserach Paper_1/(.+?).txt', eachfilename1)
        filename1=filename1.group(1)
        filename2=re.search('/GroundTruth/(.+?).txt', eachfilename1)
        filename2=filename2.group(1)
        df2= pd.read_csv('Data/' +filename1+'.txt', sep='\n',header=None)
        df2['File_Name']=filename2
        if i==0:
            df6=df2
        else:
            df6=pd.concat([df6,df2])
        i+=1
    return df6
def rouge_1_Precision(no_keyphrases,file):

    dict1={}
    tweet=[]
    totaltweetfile=0.0
    df3=dataframe(file)
    Method=[]
    with open("Data/Keyphrases_Required1/finaloutput_100thesis1.csv", 'r') as infile:
        read=csv.reader(infile, delimiter=',')

        for row in read:

            if len(row)>1:
                Method.append(row[0])
            if len(Method)==15:
                Method.remove('Method_Name')
                break
    with open("Data/Keyphrases_Required1/finaloutput_100thesis1.csv", 'r') as infile:
        read=csv.reader(infile, delimiter=',')

        for row in read:

            if len(row)==0 or 'Method_Name' in row:
                continue
            elif len(row)==1:
                tweetfilename=row[0]
                tweet.append(tweetfilename)
                totaltweetfile+=1.0
                continue
            else:
                #a=[]

                res = df3.loc[df3['File_Name'] ==tweetfilename].values.tolist()

                if no_keyphrases<=len(row):
                    n=no_keyphrases
                else:
                    n=len(row)
                data=[]
                for j in range(0,len(res)):
                    #value=res[j][0]
                    value1=res[j][0].split(' ',2)
                    data.append(value1)
                data2=[]
                for each in data:
                    for each1 in each:
                        data2.append(each1)
                data3=set(data2)

                item2=[]
                for i in range(1,n+1):
                    item=row[i].split(',',2)
                    item = item[0].replace("'", "")
                    item=item.split(' ',20)
                    item2.append(item)

                item3=[]
                for each in item2:
                    for each1 in each:
                        item3.append(each1)
                item4=set(item3)

                rougescore=rouge_1(item4, data3)

                #m=(item4, data3,rougescore)
                        #flag=True

                    #if flag==True:
                    #   break

                dict1[(tweetfilename,row[0])]=rougescore
    '''
    dict2={}
    for eachfile in tweet:
        for (eachfile1,method) in dict1.keys():
            if eachfile==eachfile1:
                value=0.0
                if len(dict1[(eachfile,method)])>0:
                    d=dict1[(eachfile,method)]

                    value=value+len(d)
                res1 = df3.loc[df3['File_Name'] ==eachfile].values.tolist()

                dict2[(eachfile,method)]=value/len(res1)
    '''

    dict3={}
    for method in Method:
        value1=0.0
        for (eachfile,each) in dict1.keys():
            if method==each:
                value1=value1+dict1[(eachfile,each)]

        dict3[method]=value1/len(tweet)

    return dict3
filename_1=str(input("Please Enter The folder Name where all ground earth files are stored="))
dict5={}
for k in range(1,21):
    dict4=rouge_1_Precision(k,filename_1)
    dict5[k]=dict4
df4=pd.DataFrame({ key:pd. Series(value) for key, value in dict5.items() })

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
iterable = (x for x in range(1,21))
x=np.fromiter(iterable, np.int)
a=list(df4.index)

plot=['p1','p2','p3','p4','p5','p6','p7','p8','p9','p10','p11','p12','p13','p14']
for i in range(0,14):
    y=list(df4.iloc[i])
    plot[i]=plt.plot(x, y, label = a[i], linestyle="-")

plt.xticks(x)
plt.yticks(ticks=(0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65))
plt.xlabel("Number of Keyphrases")
plt.ylabel("Precision")
plt.suptitle("Precision for 100thesis Data Set")

lns = plot[0]+plot[1]+plot[2]+plot[3]+plot[4]+plot[5]+plot[6]+plot[7]+plot[8]+plot[9]+plot[10]+plot[11]+plot[12]+plot[13]
labels = [l.get_label() for l in lns]
plt.legend(lns, labels, loc='upper right', bbox_to_anchor=(0.6, -0.15))
plt.savefig('Data/Results/Rouge_1_Precision_Score for thesis100.png', dpi=300,bbox_inches='tight')
plt.show()

df4.to_csv('Data/Results/results_precision_thesis100.csv')

