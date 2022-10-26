from ast import Return
import enum
from gettext import find
from itertools import combinations_with_replacement, count
from mimetypes import guess_extension
from re import T
from sys import argv
from sys import exit
from turtle import speed, up
from textwrap import dedent #三引号不缩进
from random import randint
import numpy as np 
import math
import cmath #provide functions for complex number
from fractions import Fraction #进行分数运算
import statistics
import fractions
import random #各种分布(包括正态分布)
import zlib #压缩和解压缩
import string
import numba #加速计算 (使用@numba.jit)
import time
import matplotlib
import matplotlib.pyplot as plt #绘图
import ipywidgets #这个只能在jupyter里使用，可以有交互性体验
from scipy.stats import norm #正态分布调用

def int_to_float(a):
    pass

def round_matrix(a):
    for m in range(a.shape[0]):
        for n in range(a.shape[1]):
            a[m,n]=round(a[m,n],3)+0 #处理-0.的情况, 三位小数
    return a

def linear_addition(a,b): #done
    if a.shape==b.shape:
        c=np.zeros(a.shape)
        for i in range(a.shape[0]):
            for n in range(a.shape[1]):
                c[i,n]=a[i,n]+b[i,n]
        return c
    else:
        return print("Not able to conduct addition")

def linear_product(a,b): #done
    if a.shape[1]==b.shape[0]:
       c=np.zeros([a.shape[0], b.shape[1]])
       for i in range (a.shape[0]):
           for j in range(b.shape[1]):
                for n in range(a.shape[1]):
                    c[i,j]+=a[i,n]*b[n,j]
       return c
    else:
        return print("Not able to conduct multiplication") 

def row_rearrange(a,n): #used by elimination and rref
    #n is the operating col
    product=1
    if n!=a.shape[1]-1:
        for o in range(min(a.shape[1], a.shape[0])): 
            product*=a[o,n] #the processing column has zero
        if product!=0:
            return a 
        elif product ==0:
            above_empty=False
            for z in range (n):
                above_empty=False
                if a[z,z] ==0:
                    above_empty=True
            if a[n,n]!=0 and above_empty==False: #防止一行空着不换row
                return a
            elif a[n,n]!=0: 
                for z in range(n):
                    if a[z,z]==0:
                        temp=np.copy(a[z,:])
                        a[z,:]=a[n,:]
                        a[n,:]=temp
                        return a
            else:
                i=0
                cont=True
                if n+1<a.shape[0]:
                    while cont==True and i<=(a.shape[0]-1-n):
                        for x in range(n+1,a.shape[0]):#查找没有零的一行开头
                            if a[x,n]!=0:
                                temp_data=np.copy(a[x,:]) #这里一定要copy
                                a[x,:]=a[n,:] 
                                a[n,:]=temp_data#交换两行
                                cont=False
                                i+=1
                                return a
                            else:
                                i+=1
                #整列都是0
                if cont==True:
                    return a
    else:
        return a
            
def elimination(a):
    for n in range(min(a.shape[1], a.shape[0])): #正在操作的列以及基准
        assert n!=(a.shape[1] and a.shape[1])
        a=row_rearrange(a,n)
        for m in range(a.shape[0]): #正在操作的行
            if m<=n:
                pass
            elif a[m,n]==0.:
                pass
            else:
                a[m]=a[m]-a[m,n]*a[n]/a[n,n]
    return a

print("Elimination:")
a=np.array([[0.,2.,3.],[5.,5.,6.],[7.,19.,9.],[5,99,7],[1,8,6],[7,1,0]])
print(a)
print(elimination(a)) #不包含rank

print("-------------------")

def rref(a):
    a=elimination(a)
    for mn in range (min(int(a.shape[1]),int(a.shape[0]))): #有可能会需要a.shape[1] and a.shape[2]
        if a[mn, mn]!=0:
            a[mn]=a[mn]/a[mn,mn]
        else:
            for i in range(1,a.shape[1]-mn):
                if a[mn,mn+i]!=0:
                    b=float(a[mn,mn+i])
                    for l in range (a.shape[1]):
                        a[mn, l]=a[mn,l]/b
                    break
    for mn in range(min(a.shape[0],a.shape[1])-1): #操作的行
        if a[mn+1,mn+1]!=0:
                a[mn]=a[mn]-a[mn,mn+1]*a[mn+1]/a[mn+1,mn+1]
        else: 
            for i in range(1,a.shape[1]-mn-1):
                if a[mn+1,mn+1+i]!=0:
                    a[mn]=a[mn]-a[mn,mn+1+i]*a[mn+1]/a[mn+1,mn+1+i]
                    break
    return a

print("RREF")
a=np.array([[0.,2.,3.],[5.,5.,6.],[7.,19.,9.],[5,99,7],[1,8,6],[7,1,0]])
b=rref(a)
print(b)
print("-------------------")

def find_rank(a):
    a=rref(a)
    rank=0
    for i in range(min(a.shape[0], a.shape[1])):
        if a[i, i]!=0:
            rank+=1
        else:
            for j in range(a.shape[1]-i):
                if a[i, j]!=0:
                    rank+=1
                    break
    assert rank<=min(a.shape[0], a.shape[1])
    return rank
print("------------------------------------------------")


def augmented_matrix(a,b): #增广矩阵
    if(a.shape[0] != b.shape[0]):
        raise 'The number of rows is different'
    result = []
    for i in range(len(a)):
        row = []
        for j in range(len(a[i])):
            row.append(a[i][j])
        for j in range(len(b[i])):
            row.append(b[i][j])
        result.append(row)       
    return result

def inverse(a):
    if a.shape[1]!=a.shape[0]:
        print("Not invertible")
        pass
    else:
        i=np.eye(a.shape[0])
        i=np.array(augmented_matrix(a,i))
        a=rref(i)
        a=round_matrix(a)
    return a
a=np.array([[0.,2.,3.],[5.,5.,6.],[7.,19.,9.]])
print(inverse(a))


print("--------------------------------")

def find_determinant(a, determinant=0):  
    if a.shape[0]==a.shape[1]: #check square
        if a.shape[0]==2:
            determinant+=a[0,0]*a[1,1]-a[0,1]*a[1,0]
        else: #余子式
            for i in range((a.shape[0])):
                if i==0:
                    b=a[1:,i+1:]
                    determinant+=(-1)**(i)*a[0,i]*find_determinant(b)
                else:
                    a_temp_1=a[1:, :i]
                    a_temp_2=a[1:, i+1:]
                    b=np.array(augmented_matrix(a_temp_1, a_temp_2))
                    determinant+=(-1)**(i)*a[0,i]*find_determinant(b)
    else:
        print("There's no determinant")
    return determinant
print(f"The determinant is: {find_determinant(np.array([[1,2,3],[3,5,10],[1,9,3]]))}")

print("--------------------------------")
def least_squares(a,plot=False): #这里的a是一个tuple
    x_coordinate=[]
    y_coordinate=[]
    for coordinates in a:
        x_coordinate.append(coordinates[0])
        y_coordinate.append(coordinates[1])
    #b+kx=y
    part_1=np.array([[1]*len(a)]).T
    part_2=np.array([x_coordinate]).T
    A=np.array(augmented_matrix(part_1, part_2))
    b=np.array([y_coordinate]).T
    #判断是否Ax=b可解
    ATA=np.array(linear_product(A.T, A))
    ATb=np.array(linear_product(A.T, b))
    #find ATAx_bar=ATb
    c=np.array(augmented_matrix(ATA, ATb))
    c=rref(c)
    k_bar=c[1,-1]
    b_bar=c[0,-1]
    print(f"y≈{round(k_bar,3)}x+{round(b_bar,3)}, the line do not pass through all {len(x_coordinate)} points.")
    if plot==True: #生成拟合图片
        pass
    return k_bar, b_bar
a=[(1,1),(2,2),(3,2)]
print(least_squares(a))


#find nullspace for ax=0

def find_nullspace(a):
    pivot_column=[]
    rank=find_rank(a)
    if rank==a.shape[1]:
        nullspace=0
    else:
        a=rref(a)
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                if a[i, j]==1:
                    pivot_column.append(j)
                    break
        assert rank==len(pivot_column)
        n=0
        for items in (pivot_column):
            if items==1:
                pass
            else:
                temp=np.copy(a[:,n])
                a[:,n]=a[:, items]
                a[:,items]=temp
            n+=1
        f=a[0:rank, rank:a.shape[1]]
        for i in range(f.shape[0]):
            for j in range(f.shape[1]):
                f[i,j]=-f[i,j]
        N=[]
        m=0
        while m<(a.shape[1]):
            N.append(np.arange(f.shape[1]))
            m+=1
        N=np.array(N)
        for i in range (a.shape[1]):
            for j in range (a.shape[1]-rank):
                if i<f.shape[0]:
                    N[i,j]=f[i,j]
                else:
                    if (i-rank) ==j:
                        N[i,j]=1
                    elif (i-rank)!=j:
                        N[i,j]=0
        n=0
        for items in (pivot_column):
            if items==1:
                pass
            else:
                temp=np.copy(N[n,:])
                N[n,:]=N[items,:]
                N[items,:]=temp
            n+=1

        for m in range (N.shape[1]):
            if m==0:
                print(f"c{m+1}{N[:,m]}^T")
            else:
                print(f"+c{m+1}{N[:,m]}^T")
find_nullspace(np.array([[1, 2, 2, 2], [2,4, 6, 8], [3, 6, 8, 10.]]))

print(find_determinant(np.array([[1, 27, 3], [2,24, 2], [3, 32, 3.]])))


def find_complete(a,b): #complete=particular+null 
    #b竖着输入
    rank=find_rank(a)
    if a.shape[0]!=b.shape[0]:
        print("Not able to solve")
    elif rank!=a.shape[0]:
        print("Not able to solve")
    else:
        c=augmented_matrix(a,b)
        c=elimination(c)
        print(c)
a=np.array([[1,2,3],[3,4,5],[7,8,9]])
b=np.array([[1],[2],[3]])
find_complete(a,b)
