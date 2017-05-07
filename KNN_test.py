import numpy as np
import pandas as pd
if __name__=='__main__':

    Clean_train_doc=np.load('Clean_train_doc.npy')
    sentiment_train=np.load('sentiment_train.npy')
    Clean_test_doc =np.load('Clean_test_doc.npy') 
    L=len(Clean_train_doc)

    Train_doc=Clean_train_doc
    Test_doc=Clean_test_doc
    Train_label=sentiment_train

    
    Senti_dic={}
    i=0
    for doc,s in zip(Train_doc,Train_label):
        i+=1
        if i%1000==0:
            print('Trained '+str(i)+' samples')
        if len(doc)==1:
            Senti_dic[doc[0]]=s
        elif len(doc)==2:
            if doc[0] not in Senti_dic:
                Senti_dic[doc[0]]=s
            if doc[1] not in Senti_dic:
                Senti_dic[doc[1]]=s
    print('Senti_dic constructed')

    W0=4.5
    W1=4.5
    W2=1.9
    Predict=[]
    for doc in Test_doc:
        Count=np.asarray([0,0,0,0,0])
        for word in doc:
            if word in Senti_dic:
                Temp_sent=Senti_dic[word]
                if Temp_sent==0 or Temp_sent==4:
                    Count[Temp_sent]+=W0
                elif Temp_sent==1 or Temp_sent==3:
                    Count[Temp_sent]+=W1
                elif Temp_sent==2:
                    Count[Temp_sent]+=W2
        if np.max(Count)==0:
            Temp_s=2
        else:
            Temp_s=np.argmax(Count)
        Predict.append(Temp_s)

    PhraseId=[i for i in range(156061,222353)]
    Dic={'PhraseId':PhraseId,'Sentiment':Predict}
    DF=pd.DataFrame(Dic)
    DF=DF[['PhraseId','Sentiment']]
    print(DF[:10])
    DF.to_csv('Pred.csv',index = False)
