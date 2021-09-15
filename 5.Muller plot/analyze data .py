#!/usr/bin/env python

#!/usr/bin/env python
import numpy as np
import math
import csv
import random
from scipy import stats
from scipy.optimize import curve_fit
import seaborn as sns
import matplotlib
from matplotlib import ticker
import matplotlib.pyplot as plt
from matplotlib import colors   # define the colour by myself
import matplotlib.backends.backend_pdf
import pandas as pd    # Applied to cell pushing
import time     # calculate the spend time of code
from collections import Counter     # count the frequency of mutation{{}
from itertools import combinations


df = pd.read_csv("12muller_number.csv",)
point1 = np.array(df)

G = 70
M = 500

f = open('1info_data.csv', 'a', encoding='utf - 8')
f.write('Generation,  ')
f.write('Identity,  ')
f.write('Population\n')
for i in range(1,G+1):
    for j in range(1,M+1):
        a= point1[i-1:i, j:j+1][0][0]
        print(a)
        f.write('{}, '.format(i))
        f.write('{}, '.format(j))
        if a==None:
            f.write('{},\n'.format(0))
        else:
            f.write('{},\n'.format(a))

point1[1:2,2:3]

f.close()
print('aa',point1[i:i,j:j],point1[1:2,2:3])


df = pd.read_csv("mutation 1+16.csv",)
point2 = np.array(df)
row = len(point2[0,:])
col = len(point2[:,0])
print('ww',row,col)
GG= 100
MM =18
f = open('1parent_data.csv', 'a', encoding='utf - 8')
f.write('Parent,  ')
f.write('Identity\n')
for i in range(1,col):
    for j in range(1,row):
        a= point2[i-1:i, j-1:j][0][0]
        b = point2[i-1:i, j:j+1][0][0]
        f.write('{}, '.format(a))
        f.write('{},\n'.format(b))

f.close()

data = pd.read_csv('1parent_data.csv')  # 读取csv文件

dateMap = []

for i in range(len(data)):
    dateMap.append(data["Parent"][i])

print("Quantity before removing duplicates：" + len(data).__str__())
formatList = list(set(dateMap))
formatList.sort(key=dateMap.index)

print("Quantity after removing duplicates：" + len(formatList).__str__())


print('bb',point1[1])
print('bb',point1.ix[2])
xv=[0,1,2,3,4,5,6]
label = [0,0.03125,0.0625,0.125,0.25,0.5,1]

fig = plt.figure('WHOLE_boxplot', figsize=[6, 5])
ax = fig.add_subplot(111)
all_point = (point1[6], point1[5], point1[4], point1[3], point1[2], point1[1], point1[0])

print('gg',all_point)
bplot1 = sns.violinplot(data=all_point, notch=True, vert=True, patch_artist=True,  # vertical box alignment
                         color='red', linewidth=1)
plt.ylabel("TMB Gap")
plt.xlabel("Push rate")
plt.setp(ax.collections, alpha=.6)
plt.xticks(np.arange(7), label)
ax.set_xscale("log", basex=2)


# second plot 2*2 shown the each ID
fig = plt.figure('SPLIT violinplot', figsize=[12, 10])
ax = fig.add_subplot(221)
all_point = (point1['A'], point2['A'], point3['A'], point4['A'], point5['A'], point6['A'], point7['A'])
bplot1 = sns.violinplot(xv=xv,data=all_point, notch=True, vert=True, patch_artist=True,  # vertical box alignment
                         color='red', linewidth=1)
plt.ylabel("Percentage of whole tissue")
plt.setp(ax.collections, alpha=.6)
plt.xticks(np.arange(7), label)
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))

ax = fig.add_subplot(222)
all_point = (point1['B'], point2['B'], point3['B'], point4['B'], point5['B'], point6['B'],point7['B'])
bplot1 = sns.violinplot(data=all_point, notch=True, vert=True, patch_artist=True,  # vertical box alignment
                          color='red', linewidth=1)
plt.xticks(np.arange(7), label)
plt.setp(ax.collections, alpha=.6)
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))

ax = fig.add_subplot(223)
all_point = (point1['C'], point2['C'], point3['C'], point4['C'], point5['C'], point6['C'], point7['C'])
bplot1 = sns.violinplot(data=all_point, notch=True, vert=True, patch_artist=True,  # vertical box alignment
                         color='red', linewidth=1)
plt.xlabel('Push rate')
plt.setp(ax.collections, alpha=.6)
plt.xticks(np.arange(7), label)
plt.ylabel("Percentage of whole tissue")
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))

ax = fig.add_subplot(224)
all_point = (point1['D'], point2['D'], point3['D'], point4['D'], point5['D'], point6['D'], point7['D'])
bplot1 = sns.violinplot(data=all_point, notch=True, vert=True, patch_artist=True,  # vertical box alignment
                         color='red', linewidth=1)
plt.xticks(np.arange(7), label)
plt.setp(ax.collections, alpha=.6)
plt.xlabel('Push rate')
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))



# third plot shown the mixed ID
fig = plt.figure('whole violinplot', figsize=[6, 6])
ax = fig.add_subplot(111)
all_point = (point1['A'], point2['A'], point3['A'], point4['A'], point5['A'], point6['A'], point7['A'])
bplot1 = sns.violinplot(xv=xv,data=all_point, notch=True, vert=True, patch_artist=True,  # vertical box alignment
                         color='yellow', inner='quartile', linewidth=0.6,)

all_point = (point1['B'], point2['B'], point3['B'], point4['B'], point5['B'], point6['B'],point7['B'])
bplot1 = sns.violinplot(data=all_point, notch=True, vert=True, patch_artist=True,  # vertical box alignment
                          color='darkviolet',inner='quartile', linewidth=0.8,alpha=0.1)

all_point = (point1['C'], point2['C'], point3['C'], point4['C'], point5['C'], point6['C'], point7['C'])
bplot1 = sns.violinplot(data=all_point, notch=True, vert=True, patch_artist=True,  # vertical box alignment
                         color='deepskyblue',inner='quartile',lable='ID 200', linewidth=1)

all_point = (point1['D'], point2['D'], point3['D'], point4['D'], point5['D'], point6['D'], point7['D'])
bplot1 = sns.violinplot(data=all_point, notch=True, vert=True, patch_artist=True,  # vertical box alignment
                         color='red',lable='ID 500',inner='quartile', linewidth=1.2)

# sns.boxenplot(data=all_point, color="red", width=0.05)

plt.xlabel('Push rate')
plt.xticks(np.arange(7), label)
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
plt.ylabel("Percentage of whole tissue")
plt.legend(title= '',loc='top left',labels=["Mutation ID 20","Mutation ID 100"],ncol=2)
plt.setp(ax.collections, alpha=.4)

pdf = matplotlib.backends.backend_pdf.PdfPages('Violin %s.pdf')
for fig in range(1, 2):  ## will open an empty extra figure :(
    pdf.savefig(fig)
pdf.close()
plt.show()