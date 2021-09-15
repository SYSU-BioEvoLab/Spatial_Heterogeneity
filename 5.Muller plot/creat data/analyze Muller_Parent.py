#!/usr/bin/env python
import numpy as np
import math
import csv
import pandas as pd    # Applied to cell pushing


df = pd.read_csv("data_muller_mutation.csv",)
df = df.replace(r'[\W]','',regex=True)
point2 = np.array(df)
row = len(point2[0,:])
col = len(point2[:,0])
print('ww',row,col)
f = open('parent_data.csv', 'a', encoding='utf - 8')
f.write('Parent,  ')
f.write('Identity\n')
for i in range(1,col):
    for j in range(1,row):
        a= point2[i-1:i, j-1:j][0][0]
        b = point2[i-1:i, j:j+1][0][0]
        f.write('{}, '.format(int(a)+1))
        f.write('{},\n'.format(int(b)+1))

f.close()
print('Muller_Parent.csv  File conversion completed')

data = pd.read_csv('parent_data.csv')  # 读取csv文件

dateMap = []

for i in range(len(data)):
    dateMap.append(data["Parent"][i])

print("Quantity before removing duplicates：" + len(data).__str__())
formatList = list(set(dateMap))
formatList.sort(key=dateMap.index)

print("Quantity after removing duplicates：" + len(formatList).__str__())
