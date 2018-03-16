from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
mnist=input_data.read_data_sets("MNIST_data",one_hot=True)
X_train=mnist.train.images
y_train=mnist.train.labels
X_test=mnist.test.images
y_test=mnist.test.labels

class NeuralNetwork(object):
    def __init__(self,m,t,eta,alpha,decrease_const,l1,l2,mini_batch,epochs,h=30):
        self.m=m
        self.h=h
        self.t=t
        self.eta=eta
        self.alpha=alpha
        self.decrease_const=decrease_const
        self.l1=l1
        self.l2=l2
        self.mini_batch=mini_batch
        self.epochs=epochs
        self.W1,self.W2=self.initialize_W()

    def initialize_W(self):
        W1=np.random.uniform(-1,1,size=(self.h,self.m+1))
        W2= np.random.uniform(-1, 1, size=(self.t, self.h + 1))
        return W1,W2

    def sigmoid(self,z):
        phi_z=1/(1+np.exp(-z))
        return phi_z

    def add_bias_unit(self,mat,option):
        if option=='row':
            new_mat=np.zeros((mat.shape[0]+1,mat.shape[1]))
            new_mat[0,:]=1
            new_mat[1:,:]=mat
        if option=='col':
            new_mat=np.zeros((mat.shape[0],mat.shape[1]+1))
            new_mat[:,0]=1
            new_mat[:,1:]=mat
        return new_mat

    def forward_propagation(self,w1,w2,X_train):
        a1=self.add_bias_unit(X_train,'col') # n*(m+1)
        z2=w1.dot(a1.T)  # h*n
        a2=self.sigmoid(z2)  # h*n
        a2=self.add_bias_unit(a2,'row') # (h+1)*n
        z3=w2.dot(a2)  #t*n
        a3=self.sigmoid(z3)  # t*n
        return a1,z2,a2,z3,a3

    def get_cost(self,y_enc,a3,l1,l2,W1,W2):
        term1=-np.sum(y_enc*np.log(a3))
        term2=-np.sum((1-y_enc)*np.log(1-a3))
        term3=np.sum(np.square(W2)[:,1:]*l2*0.5)+np.sum(np.abs(W1)[:,1:]*l1*0.5)
        cost=term1+term2+term3
        return cost

    def get_gradient(self,a1,a2,a3,W1,W2,y_enc):
        sigma3=a3-y_enc  # t*n
        sigma2=(W2.T.dot(sigma3))*a2*(1-a2)  #(h+1)*n
        sigma2=sigma2[1:,:]  # h*n
        delta2=sigma3.dot(a2.T)  # t*(h+1)
        delta1=sigma2.dot(a1)  # h*(m+1)
        delta2[:,1:]+=W2[:,1:]*(self.l1+self.l2)
        delta1[:,1:]+=W1[:,1:]*(self.l1+self.l2)
        return delta1,delta2

    def predict(self,X_test,W1,W2):
        a1, z2, a2, z3, a3=self.forward_propagation(W1,W2,X_test)
        y_pred=np.argmax(a3,axis=0)
        return y_pred

    def fit(self,X_train,y_train):
        self.costs=[]

        y_enc=y_train.T #t*n
        for i in range(self.epochs):
            self.eta=self.eta/(1+i*self.decrease_const)
            mini=np.array_split(range(y_train.shape[0]),self.mini_batch)
            delta1_prev=np.zeros(self.W1.shape)
            delta2_prev=np.zeros(self.W2.shape)
            for idx in mini:
                a1,z2, a2, z3, a3=self.forward_propagation(self.W1,self.W2,X_train[idx,:])
                delta1,delta2=self.get_gradient(a1,a2,a3,self.W1,self.W2,y_enc[:,idx])
                cost=self.get_cost(y_enc[:,idx],a3,self.l1,self.l2,self.W1,self.W2)
                self.costs.append(cost)
                self.W1-=(self.eta*delta1+self.alpha*delta1_prev)
                self.W2-= (self.eta * delta2 + self.alpha * delta2_prev)
                delta1_prev,delta2_prev=delta1,delta2
        return self

nn=NeuralNetwork(m=X_train.shape[1],t=10,eta=0.001,alpha=0.001,decrease_const=0.00001,l1=0,l2=0.1,mini_batch=50,epochs=500,h=50)
nn.fit(X_train,y_train)
batches=np.array_split(range(len(nn.costs)),1000)
cost_array=np.array(nn.costs)
cost_batches=[np.mean(cost_array[index]) for index in batches]
plt.plot(range(1000),cost_batches)
plt.ylabel('Cost')
plt.xlabel('Epochs')

y_pred=nn.predict(X_train,nn.W1,nn.W2)
y_true=np.argmax(y_train.T,axis=0)
accuracy=sum(y_pred==y_true)/len(y_true)
print('Train Accuracy is %f' %accuracy)

y_pred=nn.predict(X_test,nn.W1,nn.W2)
y_true=np.argmax(y_test.T,axis=0)
accuracy=sum(y_pred==y_true)/len(y_true)
print('Test Accuracy is %f' %accuracy)

misclassified_image=X_test[y_pred!=y_true][:25]
correct_label=y_true[y_pred!=y_true][:25]
misclassified_label=y_pred[y_pred!=y_true][:25]

fig,ax=plt.subplots(5,5,sharex=True,sharey=True)
ax=ax.flatten()
for i in range(25):
    img=misclassified_image[i].reshape(28,28)
    ax[i].imshow(img,cmap='Greys',interpolation='nearest')
    ax[i].set_title('t:%d,f:%d'%(correct_label[i],misclassified_label[i]))
plt.tight_layout()
plt.show()