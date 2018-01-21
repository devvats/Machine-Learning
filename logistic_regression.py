import numpy as np 
import matplotlib.pyplot as plt
def sigmoid(z):
    a = 1/(1+np.exp(-z))
    return(a)
    
def graph(x,y):
    n = np.size(x,1)
    for i in range(0,n):
        if(y[0,i]==1):
            plt.plot(x[0,i],x[1,i],'ro')
            
        else:
            plt.plot(x[0,i],x[1,i],'bo')
   
    plt.show()
           
       
def gradient(x,y,theta,alpha , iterations):
    m = np.size(x,1)
    
    x1 = np.ones((3,m))
    x1[1:,:]=x
    for i in range(0,iterations):
        theta = theta - (alpha/m)*np.dot(sigmoid(np.dot(theta,x1))-y,np.transpose(x1))
    return(theta)

def cost_function(x,y,theta):
    m = np.size(x,1)
   
    x1 = np.ones((3,m))
    
    j = (-1/m)*(np.sum(np.dot(np.log(sigmoid(np.dot(theta,x1))),np.transpose(y))+(np.dot(np.log(np.ones((1,m))-(sigmoid(np.dot(theta,x1)))),np.transpose(np.ones((1,m))-y)))))
    return(j)
    
def main():
    x = np.array([[1,2,3,4,5],[2,4,6,8,10]])
    y = np.array([[1,1,1,0,0]])
    graph(x,y)
    alpha = 0.003
    iterations = 1000
    initial_theta = np.array([1,1,1])
    print(cost_function(x,y,initial_theta))
    theta = gradient(x,y,initial_theta , alpha ,iterations)
    print(theta)
    print(cost_function(x,y,theta))
    m = np.size(x,1)
    
   
    
    
main()