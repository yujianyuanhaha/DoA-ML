#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 15:25:36 2019

@author: Jet
"""

import numpy as np



def myStack(A, stackNum):
    
    # e.g 1000*16*924 -> 1000*(16*stackNum)*924
    # A should be 2D
    len1 = np.shape(A)
    N = len1[0]
    M = len1[1]
    out = np.zeros([N,M*stackNum])
    for i in range(0,N):
        space = int((stackNum-1)/2)  # assume stackNum as odd
        temp2 = A[np.remainder(i-space+N,N),:]
        for j in range(i-space+N+1,i+space+N+1):
            temp2 = np.concatenate((temp2,A[np.remainder(j,N),:]),axis=None)
        temp2 = (temp2 - np.mean(temp2))/(np.max(temp2)-np.min(temp2))
        out[i,:] = temp2
    return out


def myStack2(A, stackNum):
    
    # e.g 1000*15*925 -> 1000*15*925*stackNum
    # A should be 3D
    len1 = np.shape(A)
    N = len1[0]
    M = len1[1]
    L = len1[2]
    out = np.zeros([N,M,L,stackNum])
    for i in range(0,N):
        space = int((stackNum-1)/2)  # assume stackNum as odd
        temp2 = A[np.remainder(i-space+N,N),:,:]
        temp2 = temp2[:,:,np.newaxis]
        for j in range(i-space+N+1,i+space+N+1):
            temp3 = A[np.remainder(j,N),:,:]
            temp3 = temp3[:,:,np.newaxis]
            temp2 = np.concatenate((temp2,temp3),axis=2)
        temp2 = (temp2 - np.mean(temp2))/(np.max(temp2)-np.min(temp2))
        out[i,:,:,:] = temp2
    return out



def myCompress(A):
    
    # 1000*16*924 -> 1000*(272*stackNum)*1
    
    temp = np.array(A)
    temp = np.matmul(temp,np.conjugate(temp.T))  #?
    temp = np.tril(temp)
    temp = temp[np.nonzero(temp)]    
    
    #temp = np.concatenate((np.real(temp),np.imag(temp)),axis=None)
    # ----- change to save angle only ------
    temp = np.angle(temp)
    
    
    temp = (temp - np.mean(temp))/(np.max(temp)-np.min(temp))
    return temp


def myCompress1(A):
    
    # 1000*16*924 -> 1000*(272*stackNum)
    len1 = np.shape(A)
    out = np.zeros([len1[0],len1[1]*(len1[1]+1)])
#    out = np.zeros([len1[0],int(len1[1]*(len1[1]+1)/2)])
    
    for i in range(0,len1[0]):    
        temp = np.array(A[i,:,:])
        temp = np.matmul(temp,np.conjugate(temp.T))  #?
        temp = np.tril(temp)
        temp = temp[np.nonzero(temp)]
        temp = np.concatenate((np.real(temp),np.imag(temp)),axis=None)
#        temp = np.angle(temp)
        temp = np.concatenate((np.absolute(temp),np.angle(temp)),axis=None)
        temp = (temp - np.mean(temp))/(np.max(temp)-np.min(temp))
        out[i,:] = temp
    return out



def myCompress2(A, stackNum):
    
    # 1000*16*924 -> 1000*(272*stackNum)
    
    temp = np.array(A)
    len1 = np.shape(A)
    N = len1[0]
    M = len1[1]
    out = np.zeros([int(N/stackNum),M*(M+1)*stackNum])
    temp2 = np.zeros([1,M*(M+1)*stackNum])
    for i in range(0,int(N/stackNum)):
        temp3 = A[i*stackNum:(i+1)*stackNum,:,:]
        for j in range(0,stackNum) :
            temp = temp3[j,:,:]
            temp = np.matmul(temp,np.conjugate(temp.T))  #?
            temp = np.tril(temp)
            temp = temp[np.nonzero(temp)]
            temp = np.concatenate((np.real(temp),np.imag(temp)),axis=None)
            temp2[:,j*M*(M+1):(j+1)*M*(M+1)] = temp
        temp2 = (temp2 - np.mean(temp2))/(np.max(temp2)-np.min(temp2))
        out[i,:] = temp2
    return out
    

def myAddnNorm(A,B, isCompress):
    
    # e.g 800*272*1 800*16*1 compress
    # e.g. 800*16*1 800*16*1848
    if isCompress:  
        temp = np.concatenate((A,B),axis=1)            
        temp = temp[:,:,0]
        [l1,l2] = np.shape(temp)
        for i in range(l1):
            temp2 = temp[i,:]
            temp[i,:] = (temp2 - np.mean(temp2))/(np.max(temp2)-np.min(temp2))
        temp = temp[:,:,np.newaxis]                
    else:
        temp = np.concatenate((A,B),axis=2)

        [l1,l2,l3] = np.shape(temp)
        for i in range(l1):
             for j in range(l2):
                temp2 = temp[i,j,:]
                temp[i,j,:] = (temp2 - np.mean(temp2))/(np.max(temp2)-np.min(temp2))
    
    return temp

if __name__ == '__main__':
    a = np.array([[1,2,3],[4,5,6]])
    b = myCompress(a)
    print(b)