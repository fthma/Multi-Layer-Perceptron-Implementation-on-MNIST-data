#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt
import numpy as np


TrainData, TrainLabel = loadlocal_mnist(
        images_path='train-images.idx3-ubyte', 
        labels_path='train-labels.idx1-ubyte')
TestData, TestLabel = loadlocal_mnist(
        images_path='t10k-images.idx3-ubyte', 
        labels_path='t10k-labels.idx1-ubyte')

#To read a subset of the dataset
def getData(Data,Labels,num):
    X=Data[500:500+num]
    D=Labels[500:500+num]
    return np.asarray(X),np.asarray(D)

#getting 10000 trainig data and 3000 testing data
X,Labels=getData(TrainData,TrainLabel,10000)
Test_X_100,Test_Labels_100=getData(TestData,TestLabel,3000)  


#to create Label arrays in desried output
def createLabelArray(Labels):
    D=[]
    for i in Labels:
        d=np.zeros(10,order='F')
        d[i]=1
        D.append(d)
    D=np.array(D)
    D=np.transpose(D)
    return D

D=createLabelArray(Labels)
TestD=createLabelArray(Test_Labels_100)


#Normalizing 10000 training data and 3000 testing data
x_train=X/255
x_test=Test_X_100/255

#Training the dataset
def MLP_Training(Xi,Dj,Beta=0.0005,k=10):
    
    Ebselon=0.015
    num=len(Xi)
    
    print("Traning begins")
    
    #initialize the weights
    U=np.random.random_sample((k,784))/25
    W=np.random.random_sample((10,k))/25
    
    Prev_DeltaU=np.zeros((k,784))
    Prev_DeltaW=np.zeros((10,k))
    
    initU=U
    initW=W
    
    X_Trans=np.transpose(Xi)
    MSE=[]
    mse=1
    
    i=0
    while(i<50):
    #while(mse>0.01): #mse>0.0001
        NetK=np.matmul(U,X_Trans)
        Zk=1/(1+np.exp(-NetK))
        Vj=np.matmul(W,Zk)
        Yj=1/(1+np.exp(-Vj))
        
    
        #backpath training
        
        Y_Trans=np.transpose(Yj)
        DeltaJ=Dj-Yj
        ejMiddle=np.matmul(Y_Trans,(1-Yj))
        Ej=np.matmul(DeltaJ,ejMiddle)
    
        W_Trans=np.transpose(W)
        DeltaK=np.matmul(W_Trans,DeltaJ)
        Zk_Trans=np.transpose(Zk)
        ekMiddle=np.matmul(Zk_Trans,(1-Zk))
        Ek=np.matmul(DeltaK,ekMiddle)
    
        DeltaW=Beta*np.matmul(Ej,Zk_Trans)
        DeltaU=Beta*np.matmul(Ek,Xi)
    
        #New_W= W + (1-Beta)*DeltaW + Beta*Prev_DeltaW
        #New_U= U + (1-Beta)*DeltaU + Beta*Prev_DeltaU
        
        #New_W= W + Beta*DeltaW + Beta*Prev_DeltaW 
        #New_U= U + Beta*DeltaU + Beta*Prev_DeltaU   
        
        New_W= W + Beta*DeltaW 
        New_U= U + Beta*DeltaU 
    
        mse=np.power(DeltaJ,2).mean()
        if((i!=0) and (mse>MSE[-1])):
            return MSE,U,W
        
        
        MSE.append(mse)
        
        U=New_U
        W=New_W
        Prev_DeltaU=DeltaU
        Prev_DeltaW=DeltaW
        
        if((i%10)==0):
            print('i:{0}.mse:{1} '.format(i,mse))
        i=i+1

    print(i)
    return MSE,U,W

#testing the data
def testing(X,D,U,W):
    X_Trans=np.transpose(X)
    num=len(X)
    X_Trans=X_Trans.reshape(784,num)
       
    print("Testing begins")
    NetK=np.matmul(U,X_Trans)
    Zk=1/(1+np.exp(-NetK))
    Vj=np.matmul(W,Zk)
    Y=1/(1+np.exp(-Vj))

    ErrorCount=np.zeros(10)
    ExactDigitOccurence=[]
    
    print("Inside Testing  loop")
    correctValueIndex=np.argmax(D,axis=0)
    predictedValueIndex=np.argmax(Y,axis=0)
    
    error=correctValueIndex-predictedValueIndex
    for i in range(len(error)):
        if(error[i]!=0):
            digit=correctValueIndex[i]
            ErrorCount[digit]=ErrorCount[digit]+1
    for i in range(10):
        ExactDigitOccurence.append(np.count_nonzero(D[i]==1))
        
                    
    PercentageError=np.asarray(ErrorCount)/np.asarray(ExactDigitOccurence)*100

    return PercentageError
    
#Drawing the bar charts
def draw_barCharts(x_train,D,x_test,TestD,K=10):
    Beta1=0.005#0.5
    Beta2=0.0005#0.05
    Beta3=0.00025#0.005
    
    #learning rate beta=0.5
    MSE,U,W=MLP_Training(x_train,D,Beta=Beta1,k=K)
    PercentageError1=testing(x_test,TestD,U,W)
    
    Label1="learningRate={}".format(Beta1)
    plt.bar(np.arange(10),PercentageError1,label=Label1, color='g')
    plt.xlabel('Digits')
    plt.ylabel('Percentage error')
    plt.legend()
    plt.title('Percentage Error for Learning Rate={}'.format(Beta1))
    plt.show()
    
    #learning rate beta=0.05
    MSE,U,W=MLP_Training(x_train,D,Beta=Beta2,k=K)
    PercentageError2=testing(x_test,TestD,U,W)
    
    Label2="learningRate={}".format(Beta2)
    plt.bar(np.arange(10),PercentageError2,label=Label2, color='g')
    plt.xlabel('Digits')
    plt.ylabel('Percentage error')
    plt.legend()
    plt.title('Percentage Error for Learning Rate={}'.format(Beta2))
    plt.show()
    
    #learning rate beta=0.025
    MSE,U,W=MLP_Training(x_train,D,Beta=Beta3,k=K)
    PercentageError3=testing(x_test,TestD,U,W)
      
    Label3="learningRate={}".format(Beta3)
    plt.bar(np.arange(10),PercentageError3,label=Label3, color='g')
    plt.xlabel('Digits')
    plt.ylabel('Percentage error')
    plt.legend()
    plt.title('Percentage Error for Learning Rate={}'.format(Beta3))
    plt.show()

#plotting the MSE Against iterations
MSE,U,W=MLP_Training(x_train,D,k=10)
plt.plot(np.arange(len(MSE)),MSE)
plt.xlabel('iterations')
plt.ylabel('Mean Square Error ')
plt.title('Mean Square Error Vs iterations')
plt.show()


#task 1
print("---------------------------------------------------")
print("TASK-1: Training-10,000, Testing-3000\n")
draw_barCharts(x_train,D,x_test,TestD,10)


#task2 with k=35
print("---------------------------------------------------")
print("TASK-2.1: k=35 Training-10,000, Testing-3000\n")
draw_barCharts(x_train,D,x_test,TestD,35)

#task2 with k=100
print("---------------------------------------------------")
print("TASK-2.2: k=100 Training-10,000, Testing-3000\n")
draw_barCharts(x_train,D,x_test,TestD,100)

#task2 with k=300
print("---------------------------------------------------")
print("TASK-2.3: k=300 Training-10,000, Testing-3000\n")
draw_barCharts(x_train,D,x_test,TestD,300)


# In[498]:





# In[732]:





# In[ ]:





# In[739]:





# In[ ]:





# In[ ]:





# In[744]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




