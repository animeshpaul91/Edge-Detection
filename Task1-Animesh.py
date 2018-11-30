# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 16:07:45 2018
@author: Animesh
"""
import cv2
import numpy as np
import math as mat

sx = [[1,0,-1], [2,0,-2], [1,0,-1]] # Sobel operator for vertical edge detection
sy = [[1,2,1],[0,0,0],[-1,-2,-1]] # Sobel operator for horizontal edge detection

def imgread():
  image = cv2.imread("task1.png", 0) #Read Image as Numpy Array
  a = []
  for i in range(0,len(image)):
      a.append(image[i])
  return a

def imgdisplay(img):
    img = np.asarray(img)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL) # Create Image Display Window
    cv2.imshow('image', img) # Display Image
    cv2.waitKey(0)
    cv2.destroyAllWindows() #Destroy Window
    
def flip(operator):
    flipped = [[0,0,0], [0,0,0], [0,0,0]]
    for i in range(0,len(operator)):
        for j in range(0,len(operator)):
            flipped[i][j] = operator[len(operator)-i-1][len(operator)-j-1] #Flip Operator 
    return flipped

def pad(m,n,img):
    temp = [[0 for x in range(n)] for y in range(m)]
    for i in range(0,len(img)):
        for j in range(0,len(img[i])): #Zero Padded Array with m+2 rows and n+2 column
            temp[i+1][j+1] = img[i][j] # Temp holds the complete padded array list. We shall use this for convolution. 
    return temp

def convolve(m,n,temp,operator):
    conv = [[0 for x in range(n)] for y in range(m)]
    for i in range(0,len(conv)):
        for j in range(0,len(conv[i])): #Inner Product Computations
            sum=0
            for k in range(0,len(operator)):
                for l in range(0,len(operator[k])):
                    sum = sum + operator[k][l]*temp[i+k][j+l]
                conv[i][j] = sum
    return conv

def normalize1(conv): #Normalize using Method-1
    storemax = []
    storemin = []
    for i in range(0,len(conv)):
        t = max(conv[i])
        m = min(conv[i])
        storemax.append(t)
        storemin.append(m)
    imax = max(storemax)
    imin = min(storemin)
    for i in range(0,len(conv)):
        for j in range(0,len(conv[i])):
            if conv[i][j] > 255:
                conv[i][j] = 255
            else:
                conv[i][j] = (conv[i][j] - imin)/(imax - imin)
    return conv

def normalize2(conv): #Normalize using Method-2 
    storemax = []
    for i in range(0,len(conv)):
        t = max(conv[i])
        storemax.append(t)
    imax = max(storemax)
    for i in range(0,len(conv)):
        for j in range(0,len(conv[i])):
            if conv[i][j] > 255:
                conv[i][j] = 255
            else:
                conv[i][j] = abs(conv[i][j])/abs(imax)
    return conv

def combine(convx, convy): #Combines Vertical and Horizontal Edges
    h = len(convx)
    w = len(convx[0])
    conv = [[0 for x in range(w)] for y in range(h)]
    for i in range(0,h): 
        for j in range(0,w):
            conv[i][j] = mat.sqrt(convx[i][j]**2 + convy[i][j]**2)
    return conv

img = imgread() # Read Image into numpy matrix

m = len(img) + 2 #Number of Horizontal Pixels
n = len(img[0]) + 2 #Number of Vertical Pixels

temp = pad(m,n,img) #Get ZeroPadded Image Matrix

#print(np.array(temp))

gx = flip(sx) #FLip Vertical Sobel 
gy = flip(sy) #FLip Horizontal Sobel 

#print(np.array(gx))
#print(np.array(gy))

convx = convolve(m-2,n-2,temp,gx) #Convolve with Vertical Sobel Operator
convy = convolve(m-2,n-2,temp,gy) #Convolve with Horizontal Sobel Operator

#Using Method1 Normalization
#convxn = normalize1(convx)
#convyn = normalize1(convy)

#conv = combine(convxn,convyn)

#imgdisplay(convxn) #Display Vertical Edges
#imgdisplay(convyn) #Display Horizontal Edges
#imgdisplay(conv)   #Display Combination of Vertical and Horizontal Edges

#Using Method2 Normalization
convxn = normalize2(convx)
convyn = normalize2(convy)

conv = combine(convxn,convyn)

imgdisplay(convxn) #Display Vertical Edges
imgdisplay(convyn) #Display Horizontal Edges
imgdisplay(conv)   #Display Combination of Vertical and Horizontal Edges
#convxn = np.array(convxn)
#convyn = np.array(convyn)
#conv = np.array(conv)
#cv2.imwrite('Vertical_Edges.png',convxn)
#cv2.imwrite('Horizontal_Edges.png',convyn)
#cv2.imwrite('Combined_Edges.png',conv)