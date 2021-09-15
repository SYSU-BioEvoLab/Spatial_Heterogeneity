#!/usr/bin/env python
import numpy as np
import math
import csv
import pandas as pd



df = pd.read_csv("data_muller_number1+18.csv")
df = df.replace(r'[\W]','',regex=True)
point1 = np.array(df)

G = 17
M = 7

f = open('12Muller_Generation.csv', 'a', encoding='utf - 8')
f.write('Generation,  ')
f.write('Identity,  ')
f.write('Population\n')
for i in range(1,G+1):
    for j in range(1,M+1):
        a= point1[i-1:i, j:j+1][0][0]
        print(a)
        f.write('{}, '.format(i))
        f.write('{}, '.format(j))
        if pd.isnull(a) :
            f.write('{}\n'.format(0))
        else:
            f.write('{}\n'.format(a))
f.close()
print('Muller_Generation.csv  File conversion completed')


