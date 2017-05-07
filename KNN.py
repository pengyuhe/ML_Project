import numpy as np

if __name__=='__main__':

    Clean_train_doc=np.load('Clean_train_doc.npy')
    sentiment_train=np.load('sentiment_train.npy')
    
    L=len(Clean_train_doc)

    TL=int(L*0.8)

    Train_doc=Clean_train_doc[:TL]
    Test_doc=Clean_train_doc[TL:]

    Train_label=sentiment_train[:TL]
    Test_label=sentiment_train[TL:]

    
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

    Weight=[float(i)/2.0 for i in range(2,10)]
    Weight2=[float(i)/10.0 for i in range(0,20)]
    Best_Acc=0
    for W0 in Weight:
        for W1 in Weight:
            for W2 in Weight2:
                Acc=0.0
                for doc,s in zip(Test_doc,Test_label):
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
                    #Temp_s=2
                    if s==Temp_s:
                        Acc+=1.0
                    
                Acc/=len(Test_label)
                if Acc>Best_Acc:
                    Best_Acc=Acc
                    print(Acc,W0,W1,W2)

    F=open('Best_Results.txt','w')
    F.write('Best_Acc,W0,W1,W2\n')
    F.write(str(Best_Acc)+' '+str(W0)+' '+str(W1)+' '+str(W2)+'\n')
    F.close()
