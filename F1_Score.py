import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
dict1={}
df=pd.read_csv('Data/Results/results_recall_thesis100.csv',index_col=0)
df1=pd.read_csv('Data/Results/results_precision_thesis100.csv',index_col=0)
a=list(df.head(14).index)

for each in range (0,14):

    b=list(df.iloc[each])
    c=list(df1.iloc[each])
    d=[]
    for i in range(0,len(b)):
        if b[i]==0.0 and c[i]==0:
            d.append(0.0)
        else:
            d.append(2*(b[i]*c[i])/(b[i]+c[i]))
    dict1[a[each]]=d
df2=pd.DataFrame.from_dict(dict1, orient ='index')



fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
iterable = (x for x in range(1,21))
x=np.fromiter(iterable, np.int)
f=list(df2.index)

plot=['p1','p2','p3','p4','p5','p6','p7','p8','p9','p10','p11','p12','p13','p14']
for i in range(0,14):
    y=list(df2.iloc[i])
    plot[i]=plt.plot(x, y, label = f[i], linestyle="-")

plt.xticks(x)
plt.yticks(ticks=(0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65))
plt.xlabel("Number of Keyphrases")
plt.ylabel("F1")
plt.suptitle("F1 for thesis100 Data Set")

lns = plot[0]+plot[1]+plot[2]+plot[3]+plot[4]+plot[5]+plot[6]+plot[7]+plot[8]+plot[9]+plot[10]+plot[11]+plot[12]+plot[13]
labels = [l.get_label() for l in lns]
plt.legend(lns, labels, loc='upper right', bbox_to_anchor=(0.6, -0.15))
plt.savefig('Data/Results/Rouge_1_F1_Score for thesis100.png', dpi=300,bbox_inches='tight')
plt.show()




df2.to_csv('Data/Results/results_f1score_thesis100.csv')



