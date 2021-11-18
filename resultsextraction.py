import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df1=pd.read_csv('Data/Results/results_precision_Thesis100.csv',header=None)
column=[i for i in range(20)]
Row=['SingleRank EdgeGraph','PositionRank EdgeGraph','TextRank EdgeGraph','NERank EdgeGraph','SingleRank WCN','PositionRank WCN','TextRank WCN','NERank WCN']
df2=pd.DataFrame(index=Row,columns=column)
dict1={}
a=list(df1.iloc[1,:])
a.remove(a[0])
b=list(df1.iloc[3,:])
b.remove(b[0])
c=list(df1.iloc[5,:])
c.remove(c[0])
d=list(df1.iloc[6,:])
d.remove(d[0])
e=list(df1.iloc[14,:])
e.remove(e[0])
f=list(df1.iloc[9,:])
f.remove(f[0])
g=list(df1.iloc[11,:])
g.remove(g[0])
h=list(df1.iloc[12,:])
h.remove(h[0])

list1=[a,b,c,d,e,f,g,h]
for i,each in enumerate(Row):
    dict1[each]=list1[i]
df3=pd.DataFrame({ key:pd. Series(value) for key, value in dict1.items() })
df4=df3.T

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
iterable = (x for x in range(1,21))
x=np.fromiter(iterable, np.int)
a=list(df4.index)

plot=['p1','p2','p3','p4','p5','p6','p7','p8']
for i in range(0,8):
    y=list(df4.iloc[i])
    plot[i]=plt.plot(x, y, label = a[i], linestyle="-")

plt.xticks(x)
plt.yticks(ticks=(0.06,0.12,0.18,0.24,0.30))
plt.xlabel("Number of Keyphrases")
plt.ylabel("Precision")
plt.suptitle("Precision for theses100 Data Set")

lns = plot[0]+plot[1]+plot[2]+plot[3]
lns1=plot[4]+plot[5]+plot[6]+plot[7]
labels = [l.get_label() for l in lns]
labels1=[l.get_label() for l in lns1]
l1=plt.legend(lns, labels, loc='upper right', bbox_to_anchor=(0.5, -.15) )
l2=plt.legend(lns1, labels1, loc='upper right', bbox_to_anchor=(0.9, -.15))
plt.gca().add_artist(l1)
plt.gca().add_artist(l2)
plt.savefig('Data/Results_Con/precision_Thesis100111.png', dpi=300,bbox_inches='tight')
plt.show()



#df4.to_csv('/Users/Files From e.localized/Poornima Docs/Machine Learning/Dr. Muskaan Garg/Reserach Paper_1/Results_Con/Recall_wiki20.csv')

