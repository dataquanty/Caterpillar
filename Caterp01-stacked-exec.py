# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 16:29:13 2015

@author: dataquanty
"""
import glob
import pandas as pd

"""
i = 0
while i<50:
    execfile("Caterp01-stacked.py")
    i+=1
    print "Exec num " + str(i)


"""

"""
lstfiles = glob.glob('pred06-stack*')
size = 80

p1 = pd.read_csv(lstfiles[0])
for i in range(1,size):
    p2 = pd.read_csv(lstfiles[i])
    p1 = pd.concat([p1['id'],p1['cost']+p2['cost']],axis=1)

p1['cost']=p1['cost']/size
p1.to_csv('pred06-stack40.csv',index=False)

"""
