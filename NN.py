import numpy as np 
import pandas as pd


def forward(x,w1,w2,w3,b1,b2,b3):
    hl1 = np.tanh(np.dot(x,w1) + b1)
    hl2 = np.tanh(np.dot(hl1,w2)+b2)
    y = np.tanh(np.dot(hl2,w3)+b3)
    return(hl1,hl2,y)
def deltab(hl1,hl2,y,ans,w3,w2):
    s3 = np.sech(y)
    delb3 = 2*np.dot((y-ans),np.dot(np.transpose(s3),s3))
    s2 = np.sech(hl2)
    delb2 = np.dot(delb3,np.dot(np.transpose(w3),np.dot(np.transpose(s2),s2)))
    s1 = np.sech(hl1)
    delb1 = np.dot(delb2, np.dot(np.transpose(w2),np.dot(np.transpose(s1),s1)))
    delb11 = np.zeros((1,12))
    delb22 = np.zeros((1,12))
    delb33 = np.zeros((1,1))
    for i in range(12):
        delb11[0,i]= np.sum(delb1[:,i])
    for i in range(12):
        delb22[0,i]= np.sum(delb2[:,i])
    delb33[0,0]= np.sum(delb3)
    return(delb11,delb22,delb33)
    
def delta(hl1,hl2,y,ans,w2,w3):
    del4 = y-ans
    g3 = (1-hl2)*hl2
    del3 = np.dot(del4,np.transpose(w3))*g3
    g2 = (1-hl1)*hl1
    del2 = np.dot(del3,np.transpose(w2))*g2
    return(del2,del3,del4)
def tri(x,hl1,hl2,del2,del3,del4):
    tri3 = np.dot(np.transpose(hl2),del4)
    tri2 = np.dot(np.transpose(hl1),del3)
    tri1 = np.dot(np.transpose(x),del2)
    return(tri1,tri2,tri3)
def updatew(w1,w2,w3,m,tri1,tri2,tri3):
    w1  =w1-(1/m)*tri1
    w2  =w1-(1/m)*tri2
    w3 =w1-(1/m)*tri3
    return(w1,w2,w3)
def updateb(b1,b2,b3,m,delb11,delb22,delb33):
    b1 = b1-(1/m)*delb11
    b2 = b2 -(1/m)*delb22
    b3 = b3 -(1/m)*delb33
    return(b1,b2,b3)

def findwb(x,ans,w1,w2,w3,b1,b2,b3,iterations):
    m = np.size((x,0))
    for i in range(iterations):
        hl1,hl2,y = forward(x,w1,w2,b1,b2,b3)
        del11,del22,del33 = deltab(hl1,hl2,y,ans,w3,w2)
        del2,del3,del4 = delta(hl1,hl2,y,ans,w2,w3)
        tri1,tri2,tri3= tri(x,hl1,hl2,del2,del3,del4)
        w1,w2,w3 = updatew(w1,w2,w3,m,tri1,tri2,tri3)
        b1,b2,b3= updateb(b1,b2,b3,m,del11,del22,del33)
    return(w1,w2,w3,b1,b2,b3)
    

def main():
    # assumption: x is mX7 matrix with 7 features , y is mX1 , sizes od hidden layers is 12 each.
    w1 = np.random.uniform(size = (7,12))
    w2 = np.random.uniform(size = (12,12))
    w3 = np.random.uniform(size = (12,1))
    b1 = np.random.uniform(size = (1,12))
    b2 = np.random.uniform(size = (1,12))
    b3 = np.random.uniform(size = (1,1))
    