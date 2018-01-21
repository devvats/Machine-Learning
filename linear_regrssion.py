import numpy as np 
import matplotlib.pyplot as plt
def graph(x,y,line):
    for i in range(0,np.size(x,1),):
        plt.plot(x[0,i],y[0,i],'ro')
        
        
    plt.plot(x,line) 
    plt.show()
def gradient(x1,y,theta,iterations,alpha):
    m = np.size(x1,1)
    
    for i in range(0,iterations):
        theta = theta -(alpha/m)*np.dot((np.dot(theta , x1)-y),(np.transpose(x1)))
    return(theta)
def cost_function(x1, y ,theta):
    m = np.size(x1,1)
    print(np.size(x1))
    j = (1/(2*m))*np.sum(np.square(np.dot(theta,x1)-y))
    return(j)
    
    
def main():
    x = np.array([[0,1,2,3,4,5,6,]])
    y = np.array([[0,2,4,6,8,10,12,]])
    m = np.size(x)
    x1 = np.ones((2,m))
    x1[1:,:]=x
    
    alpha = 0.003
    iterations = 1000
    initial_theta = np.array([[0,0]])
    line = np.dot(initial_theta,x1)
    graph(x,y,line)
    
    print(cost_function(x1,y,initial_theta))
    theta = gradient(x1,y,initial_theta,iterations,alpha)
    print(theta)
    print(cost_function(x,y,theta))

main()