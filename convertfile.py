
import glob

import re
#import os

listoffile1=sorted((glob.glob("Data/theses100/docsutf8/*.txt")))
k=1
for eachfilename1 in listoffile1:

    #os.rename(eachfilename1,r'/Users/Files From e.localized/Poornima Docs/Machine Learning/Dr. Muskaan Garg/Reserach Paper_1/110-PT-BN-KP/keys/'+str(k)+'.txt')
    #k+=1
    dict1={}
    filename1=re.search('/docsutf8/(.+?).txt', eachfilename1)
    filename1=filename1.group(1)
    with open("Data/theses100/docsutf8/" +filename1+'.txt', 'r') as infile:
        read=infile.readlines()

        i=0
        for row in read:
            read1=row.split('.')

            for each in read1:
                if(len(each)>8):
                    dict1[i]=each
                i+=1
    dict1.popitem()
    try:
        geeky_file = open('Data/100Thesis/' +filename1+'.txt', 'wt')
        geeky_file.write(str(dict1))
        geeky_file.close()

    except:
        print("Unable to write to file")

