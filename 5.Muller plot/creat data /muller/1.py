'''
  [Muller plot of Cancer mutation model]

  [2021/7/13]
  the cell divide from the cancer stem cell, and every divide will
  create a set of new mutation, this code will save the CSV file of
  mutation ID and family tree for every single cell.
  then import the file to "analyze Muller_generation.py"
  and "analyze Muller_parent.py" to get the order file.
  In the final,import the parent and generation file into R to visualization.
'''

import numpy as np
import math
import random
from scipy import stats
import scipy.stats as st
from scipy.optimize import curve_fit
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors   # define the colour by myself
import matplotlib.backends.backend_pdf
import pandas as pd    # Applied to cell pushing
import time     # calculate the spend time of code
from collections import Counter     # count the frequency of mutation{{}
from itertools import combinations  # appy jaccard index set combian each other
start_time = time.time()

Max_generation = 18
push_rate = 1 # the frequency of cancer cell push each others
Max_ROW = 1000
Max_COL = 1000

remove_frequency = 0
# growth generation and mark mutation ID in the cancer tissue
Heatmap_gap_boxplot = True
Heatmap_whole_plot = False
Heatmap_remove_frequency = True
save_mutation = False  # 1 is save file, 0 is not
### Basic parameter setting
background_gene = 1   # The background gene that cancer originally carried
birth_rate = 1
death_rate = 0


sample_cut_size = 1  #warning only num**2 ,Size of each sample area

sampling_size = 100   # The amount of a sample taken, sample number in the tissue
Filter_frequency = 100 #filter the allele frequency less than this value.
# birth_rate = int(input('Birth rate ='))
# death_rate = int(input('death rate ='))  # The death probability of cell growth process
die_divide_times = 500   # The max times of cell divides,  arrived this point the cell will die
Poisson_lambda = 1   # (in generaral means mutation_rate in cancer reseach),the mean value of Poisson distributions


heatmap_gener = 1
mutation_heat = 0

mutation_type1 = 1
mutation_type2 = 50
mutation_type3 = 100
mutation_type4 = 200
mutation_type5 = 500
mutation_type6 = 1000

p_color = ['#E2E28A', 'k', '#80E58C', '#4E99D8', '#C1C1C1', '#E080D7','#E28181']  # same color for boxplot and jaccard index
# cancer growth space size



### Show the fellowing mutation ID in the cencer tissue
global all_mutation
global list_newcell
global all_ip
global list_mutation
global less_list
list_newcell =[]
list_newcell2 =[]

list0_table =[]
list1_table =[]
list2_table =[]
list3_table =[]
list_mutation =[]
all_mutation = []
all_mutation_del = []
all_ip =[]
less_list =[]
less_list2 =[]
less_list3 =[]
diagnal_pushing_weight = 1/2

# mutation_type5 = 55
# mutation_type6 = 0

mesh_shape = (Max_ROW, Max_COL)     # growth grid
cancer_matrix = np.zeros(mesh_shape)    # create grid space of the cancer growth
cancer_heatmap = np.zeros(mesh_shape)
filter_heatmap = np.zeros(mesh_shape)

# f = open('voilin_plot.csv', 'a', encoding='utf - 8')
# f.write('Pushrate  ')
# f.write('{}\n'.format(push_rate))
# f.close()


if save_mutation == True:
    f = open('Muller plot%s+{}.csv'.format(Max_generation) % push_rate, 'a', encoding='utf - 8')
    f.write('Slope  ')
    f.write('R^2  ')
    f.write('Ks\n')
    f.close()

def cancer_mesh(row,col):    # define cancer growth mesh size
    all_mesh = [(r,c)for r in np.arange(0,row) for c in np.arange(0, col)]
    central_pos = {cp:neighbor_mesh(cp,row,col) for cp in all_mesh}  # define the center position
    return central_pos

def neighbor_mesh(cp,row,col):  # define the neighbor of the center point.
    r, c = cp     # cp: central position   r: cp.row  c: cp.column
    neighbor_pos = [(r+i, c+j)
        for i in [-1, 0, 1]
        for j in [-1, 0, 1]
        if 0 <= r + i < row     # define the point in the grid
        if 0 <= c + j < col
        if not (j == 0 and i == 0)]
    return neighbor_pos

# Function simplification
binomial = np.random.binomial
shuffle = np.random.shuffle
randint = np.random.randint
random_choice = random.choice
poisson = np.random.poisson

def birth_r():  #death rate
    birth = binomial(1, birth_rate)
    return birth
def death_r():  #death rate
    death = binomial(1, death_rate)
    return death
def push_r():  #death rate
    pushing = binomial(1, push_rate)
    return pushing
def Poisson():   # mutation numbers from Poisson distribution, where the lambda=10
    add_mutation =poisson(Poisson_lambda)     # cancer_num: the total number of cancer cells in the times
    # change the size to 1
    return add_mutation
def Fit_Rsqr(xd, yd,order=1):
    coeffs = np.polyfit(xd, yd, order)

    intercept = coeffs[-1]
    slope = coeffs[-2]
    power = coeffs[0] if order == 2 else 0

    p = np.poly1d(coeffs)

    ybar = np.sum(yd) / len(yd)
    ssreg = np.sum((p(xd) - ybar) ** 2)
    sstot = np.sum((yd - ybar) ** 2)
    Rsqr = ssreg / sstot
    return Rsqr

def fit_func(x,a,b):  # fitting the VAF accumulation curve
    return a*(x-1/0.495)

def poisson_func(x):
    return st.poisson(x)

def line_func(x,a,b):
    return a*x+b
def Jaccard_func(x, a, b):  # fitting the jaccard curve equation
    return a*(np.power(2,x)**b)

def Fit_line(burden_size, burden_number):
    popt, pcov = curve_fit(line_func, burden_size, burden_number)

    x_line = np.array([min(burden_size), max(burden_size)])
    y_line = popt[0] * x_line + popt[1]

    # Plot trendline
    # plt.plot(xl, yl,'orange', alpha=1)
    plt.plot(x_line, y_line, 'r', )

    a = popt[0]
    b = popt[1]
    Rsqr = Fit_Rsqr(burden_size, burden_number)
    plt.text(50,0.5*max(y_line), "y=%.2f *x%.2f \n R^2= %.3f" % (a, b,Rsqr), color='r')

def trendline(xd, yd,zd, term, order=1, c='r', alpha=0.2, Rval=False):
    """Make a line of best fit"""

    popt, pcov = curve_fit(fit_func, xd, yd)
    fit_slope = popt[0]

    # yvals = fit_func(xd, a)  # 拟合y值

    Rsqr = Fit_Rsqr(xd, yd)
    #Calculate trendline

    minxd = np.min(xd)
    maxxd = np.max(xd)

    xl = np.array([minxd, maxxd])
    yl =  fit_slope * xl
    ya = fit_slope * (xl-1/0.495)
    y1 = stats.ks_2samp(zd, yl)

    #Plot trendline
    # plt.plot(xl, yl,'orange', alpha=1)
    plt.plot(xl, ya, 'b',alpha=0.1)

    #Calculate R Squared


    # Kolmogorov-Smirnov test
    Data_dic = {}
    K_distance = []
    KS_dict ={}
    for i in range(len(xd)):
        Data_dic[xd[i]] = yd[i]
    for i in xd:
        K_gap = abs(Data_dic[i] - (fit_slope * (i - 1/0.495)))
        K_distance.append(K_gap)
        KS_dict[K_gap] = i
    Ks_distance = map(abs, K_distance)
    KS=(max(Ks_distance)) / (max(yd))
    KS_pos = KS_dict[max(K_distance)]
    # ys = np.array[0,max(Ks_distance)]
    xs = KS_pos


    if not Rval:
        #Plot R^2 value
        plt.text(0.5 * maxxd + 0.2 * minxd, 1.3 * np.max(yd) + 0.3 * np.min(yd),
                 '$R^2 = %0.3f$' % Rsqr)
        plt.text(0.5 * maxxd + 0.2 * minxd, 1.2 * np.max(yd) + 0.3 * np.min(yd),
                 "S: %.2f" % slope )
        plt.text(0.5 * maxxd + 0.2 * minxd, 1.1 * np.max(yd) + 0.3 * np.min(yd),
                 "Ks: %.2f" % KS)
        plt.text(0.99*KS_pos, Data_dic[KS_pos],
                 "_ks" )

    else:
        #Return the R^2 value:
        print('slope{}'.format(term),fit_slope, '$R^2 = %0.3f' % Rsqr,'K-S',KS)
        # print('slope{}'.format(term), slope, 'K-S',KS)
        if save_mutation == True:
            f = open('Muller plot%s+{}.csv'.format(Max_generation) % push_rate, 'a', encoding='utf - 8')
            f.write('{}  '.format(round(fit_slope, 2)))
            f.write('{}  '.format(round(Rsqr, 3)))

            f.write('{}\n'.format(round(KS, 3)))
            f.close()
def Distance_KS(xd, yd):   #Kolmogorov–Smirnov test
    popt, pcov = curve_fit(fit_func, xd, yd)
    Data_dic = {}
    K_distance = []
    KS_dict ={}
    for i in range(len(xd)):
        Data_dic[xd[i]] = yd[i]
    for i in xd:
        K_gap = abs(Data_dic[i] - (Poisson_lambda * (i - 1/0.495)))
        K_distance.append(K_gap)
        KS_dict[K_gap] = i
    Ks_distance = map(abs, K_distance)
    Max_ks = max(Ks_distance)
    KS=(Max_ks) / (max(yd))
    KS_pos = KS_dict[max(K_distance)]
    plt.text(0.99 * KS_pos, Data_dic[KS_pos],"_ks")
    return Max_ks,KS
def jacard_index(set_a, set_b):
    intersection_set = list(set(set_a).intersection(set(set_b)))
    union_set = list(set(set_a).union(set(set_b)))
    jacard_index = len(intersection_set)/len(union_set)
    return jacard_index
def combine(temp_list, n):
    '''根据n获得列表中的所有可能组合（n个元素为一组）'''
    temp_list2 = []
    for c in combinations(temp_list, n):
        temp_list2.append(c)
    return temp_list2
def jacard_run(run_set,all_mutation_dic):
    combine_set = combine(run_set,2)

    X_distance = []
    Y_jaccard = []



    for i, j in combine_set:
        i_x,i_y = i
        j_x, j_y = j
        D_i_j = math.sqrt((j_x-i_x)**2+(j_y-i_y)**2)   # calculate the distance of two combination points

        X_distance.append(D_i_j)
        Y_jaccard.append(jacard_index(all_mutation_dic[i],all_mutation_dic[j]))

    return X_distance,Y_jaccard,
    #     print('assssss',i, j,D_i_j,jacard_index(all_mutation_dict1[i],all_mutation_dict3[j]), )
    # print('zzz',math.factorial(10),len(X_distance),len(Y_jaccard))


def Func_fitting_plot(N,xdata,ydata,Fit = False):
    plt.figure('ID_index {}'.format(N), figsize=[5, 5])

    plt.scatter(xdata, ydata,   s=20, alpha=0.5)
    plt.xlabel('Distance')
    plt.ylabel("Mutation number gap")
    # plt.legend(labels='', loc='upper right', shadow=bool, title='Jaccard Num= {}'.format(len(ydata)), numpoints=None)

    if Fit == True:

        popt, pcov = curve_fit(line_func, xdata, ydata)

        # ### Calculate R Square ###
        # calc_ydata = [Jaccard_func(i, popt[0], popt[1]) for i in xdata]
        # res_ydata = np.array(ydata) - np.array(calc_ydata)
        # ss_res = np.sum(res_ydata ** 2)
        # ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
        # r_squared = 1 - (ss_res / ss_tot)

        # popt, pcov = curve_fit(Jaccard_func, xdata, ydata)
        a = popt[0]
        b = popt[1]
        Rsqr = Fit_Rsqr(xdata, ydata)
        # print('a',a,b)
        x_line = np.array([min(xdata), max(xdata)])
        y_line = popt[0] * x_line + popt[1]

        plt.plot(x_line, y_line, 'r', label='polyfit values')

        plt.text(5, max(y_line), "y=%.2f *x+%.2f " % (a, b), color='r')
        plt.text(6, max(y_line)-max(y_line)/10, "R^2=%.3f " % Rsqr, color='r')


def Jacard_boxplot(cut_dis,X_distance,Y_jaccard):  #plot th boxplot for diffrent dstance 20 -100
    p1=20 #define the plot position and ±1 bar
    p2=40
    p3=60
    p4=80
    p5=100
    point = [p1,p2,p3,p4,p5]
    point1 = []
    point2 = []
    point3 = []
    point4 = []
    point5 = []
    Y0=[]
    X0=[]
    gap=5
    jacard_dict=zip(X_distance, Y_jaccard)
    for i,j in jacard_dict:  #filter jaccard point in the diffrent distance
        if i < cut_dis:
            X0.append(i)
            Y0.append(j)
        if p1-gap <= i <= p1+gap:
            point1.append(j)
        elif p2-gap <= i <= p2+gap:
            point2.append(j)
        elif p3 - gap <= i <= p3 + gap:
            point3.append(j)
        elif p4 - gap <= i <= p4 + gap:
            point4.append(j)
        elif p5 - gap <= i <= p5 + gap:
            point5.append(j)

    labels = ['20', '40', '60','80', '100',] # for X Axis lable

    point1_mean = np.mean(point1)
    point1_std = np.std(point1,ddof=1)
    fig = plt.figure('jacard_boxplot', figsize=[5, 5])
    ax = fig.add_subplot(111)
    all_point=(point1,point2,point3,point4,point5)
    bplot1 = ax.boxplot(all_point,vert=True,patch_artist = True,  # vertical box alignment
                     labels=labels)
    for box in bplot1['boxes']:
        box.set(color='k',
                linewidth=1)
    for median in bplot1['medians']:
        median.set(color='red',
                   linewidth=1.5)
    plt.xlabel('Distance')
    plt.ylabel("Jaccard index")

    # Func_fitting_plot(20,X_distance, Y_jaccard)
    Func_fitting_plot(20,X_distance,Y_jaccard)
    Func_fitting_plot(10, X0, Y0,Fit = True)

'''calculate the number of mutation in cancer cell,
compare the gap between the diffrent distance'''
def Distance_gap(set_a,set_b):
    D_gap=abs(len(set_a)-len(set_b))
    return D_gap

def ID_run(run_set, all_mutation_dic,cell_dictionary):

    X_distance = []
    Y_jaccard = []

    for i in run_set:
        i_x, i_y = i

        j_x, j_y = (Max_ROW/2,Max_COL/2)
        D_i_j = math.sqrt((j_x - i_x) ** 2 + (j_y - i_y) ** 2)  # calculate the distance of two combination points

        X_distance.append(D_i_j)
        Y_jaccard.append(Distance_gap(all_mutation_dic[i], cell_dictionary[j_x, j_y ].mutation))

    return X_distance, Y_jaccard

def Cut_area_mutation(x,y,size,cell_dictionary):
    all_cut = []
    for i in range(x - size, x + size):
        for j in range(x - size, x + size):
            all_cut.append(cell_dictionary[i,j].mutation )



def Point_distance(run_set,all_mutation_dic,cell_dictionary):
    combine_set = combine(run_set,2)
    X_distance = []
    for i, j in combine_set:
        i_x,i_y = i
        j_x, j_y = j
        D_i_j = math.sqrt((j_x-i_x)**2+(j_y-i_y)**2)   # calculate the distance of two combination points

        X_distance.append(D_i_j)
    return X_distance
# def Cut_remove(x,y,):

#2021.1.12 cur unique mutation and distance
def Uni_distance(mut_ID,point_set,all_mutation_dic,cell_dictionary):
    combine_set = combine(point_set, 2)
    X_distance = []
    Y_uni_number =[]
    for i, j in combine_set:
        i_x, i_y = i
        j_x, j_y = j
        D_i_j = math.sqrt((j_x - i_x) ** 2 + (j_y - i_y) ** 2)  # calculate the distance of two combination points

        X_distance.append(D_i_j)

        if mut_ID in all_mutation_dic[i] and mut_ID in all_mutation_dic[j]:
            Y_uni_number.append(1)
        else:
            Y_uni_number.append(0)

    gap=2.5
    p1=2.5
    p2=7.5
    p3 = 12.5
    p4 = 17.5
    p5 = 22.5
    p6 = 27.5
    p7 = 32.5
    p8 = 37.5
    p9 = 42.5
    p10 = 47.5
    p11 = 52.5
    p12 = 57.5
    p13 = 62.5
    p14 = 67.5
    p15 = 72.5
    p16 = 77.5
    p17 = 82.5
    p18 = 87.5
    p19 = 92.5
    p20 = 97.5
    point1 = []
    point2 = []
    point3 = []
    point4 = []
    point5 = []
    point6 = []
    point7 = []
    point8 = []
    point9 = []
    point10 = []
    point11 = []
    point12 = []
    point13 = []
    point14 = []
    point15 = []
    point16 = []
    point17 = []
    point18 = []
    point19 = []
    point20 = []
    unique_dict = zip(X_distance, Y_uni_number)
    for i, j in unique_dict:  # filter Mutation point in the diffrent distance

        if p1 - gap <= i < p1 + gap:
            point1.append(j)
        elif p2 - gap <= i < p2 + gap:
            point2.append(j)
        elif p3 - gap <= i < p3 + gap:
            point3.append(j)
        elif p4 - gap <= i < p4 + gap:
            point4.append(j)
        elif p5 - gap <= i < p5 + gap:
            point5.append(j)
        elif p6 - gap <= i < p6 + gap:
            point6.append(j)
        elif p7 - gap <= i < p7 + gap:
            point7.append(j)
        elif p8 - gap <= i < p8 + gap:
            point8.append(j)
        elif p9 - gap <= i < p9 + gap:
            point9.append(j)
        elif p10 - gap <= i < p10 + gap:
            point10.append(j)
        if p11 - gap <= i < p11 + gap:
            point11.append(j)
        elif p12 - gap <= i < p12 + gap:
            point12.append(j)
        elif p13 - gap <= i < p13 + gap:
            point13.append(j)
        elif p14 - gap <= i < p14 + gap:
            point14.append(j)
        elif p15 - gap <= i < p15 + gap:
            point15.append(j)
        elif p16 - gap <= i < p16 + gap:
            point16.append(j)
        elif p17 - gap <= i < p17 + gap:
            point17.append(j)
        elif p18 - gap <= i < p18 + gap:
            point18.append(j)
        elif p19 - gap <= i < p19 + gap:
            point19.append(j)
        elif p20 - gap <= i < p20 + gap:
            point20.append(j)

    labels = list(range(5,105,5))  # for X Axis lable

    point1_mean = np.mean(point1)
    point1_std = np.std(point1, ddof=1)
    fig = plt.figure('Distance_Unique_number', figsize=[5, 5])

    all_point=[sum(point1),sum(point2),sum(point3),sum(point4),sum(point5),sum(point6),sum(point7),sum(point8),sum(point9),sum(point10),
               sum(point11),sum(point12),sum(point13),sum(point14),sum(point15),sum(point16),sum(point17),sum(point18),sum(point19),sum(point20)]

    plt.bar(labels,all_point,width=3,alpha=1,label='ID:{}'.format(mut_ID))
    plt.xlabel('Distance')
    plt.legend(loc='best')
    plt.ylabel("Number Unique ID")
    plt.title('pushrate: %s, sample number: {}'.format(sampling_size) % push_rate)



def Distance_mut_num_run(run_set,all_mutation_dic,cell_dictionary):
    combine_set = combine(run_set,2)

    X_distance = []
    Y_number = []



    for i, j in combine_set:
        i_x,i_y = i
        j_x, j_y = j
        D_i_j = math.sqrt((j_x-i_x)**2+(j_y-i_y)**2)   # calculate the distance of two combination points

        X_distance.append(D_i_j)
        Y_number.append(Distance_gap(all_mutation_dic[i],all_mutation_dic[j]))

    return X_distance,Y_number

def Distance_mut_num_boxplot(cut_dis,X_distance,Y_Mutation):  #plot th boxplot for diffrent dstance 20 -100
    p1=10 #define the plot position and ±1 bar
    p2=20
    p3=30
    p4=40
    p5=50
    p6 = 60
    p7 = 70
    p8 = 80
    p9 = 90
    p10 = 100
    point = [p1,p2,p3,p4,p5]
    point1 = []
    point2 = []
    point3 = []
    point4 = []
    point5 = []
    point6 = []
    point7 = []
    point8 = []
    point9 = []
    point10 = []
    Y0=[]
    X0=[]
    gap=5
    mutation_dict=zip(X_distance, Y_Mutation)
    for i,j in mutation_dict:  #filter Mutation point in the diffrent distance

        if i < cut_dis:
            X0.append(i)
            Y0.append(j)
        if p1 - gap <= i <= p1 + gap:
            point1.append(j)
        elif p2 - gap <= i <= p2 + gap:
            point2.append(j)
        elif p3 - gap <= i <= p3 + gap:
            point3.append(j)
        elif p4 - gap <= i <= p4 + gap:
            point4.append(j)
        elif p5 - gap <= i <= p5 + gap:
            point5.append(j)
        elif p6 - gap <= i <= p6 + gap:
            point6.append(j)
        elif p7 - gap <= i <= p7 + gap:
            point7.append(j)
        elif p8 - gap <= i <= p8 + gap:
            point8.append(j)
        elif p9 - gap <= i <= p9 + gap:
            point9.append(j)
        elif p10 - gap <= i <= p10 + gap:
            point10.append(j)

    labels = [p1, p2, p3,p4, p5,p6, p7, p8,p9, p10] # for X Axis lable

    point1_mean = np.mean(point1)
    point1_std = np.std(point1,ddof=1)

    fig = plt.figure('Distance_gap_boxplot %s' % cut_dis, figsize=[5, 5])
    ax = fig.add_subplot(111)
    all_point=(point1,point2,point3,point4,point5,point6,point7,point8,point9,point10)
    if cut_dis == 1:
        bplot1 = sns.violinplot(data= all_point,notch=True,vert=True,patch_artist = True,  # vertical box alignment
                     xlabels=labels,color='red',linewidth=1)
    if cut_dis == 2:
        bplot1 = ax.boxplot(all_point, vert=True, patch_artist=True,  # vertical box alignment
                            labels=labels)
        for box in bplot1['boxes']:
            box.set(color='skyblue',
                    linewidth=1)
        for median in bplot1['medians']:
            median.set(color='red',
                       linewidth=1.5)
    plt.setp(ax.collections, alpha=.6)
    plt.xlabel('Distance')
    plt.ylabel("Number variance")
    plt.title('pushrate: %s, filter: {}'.format(remove_frequency) % push_rate)

    # Func_fitting_plot(20,X_distance, Y_Mutation)

    if cut_dis ==1 :
        Func_fitting_plot(cut_dis, X_distance, Y_Mutation, Fit=True)

    else:
        Func_fitting_plot(cut_dis, X_distance, Y_Mutation)

    f = open('voilin_plot%s+{}.csv'.format(Max_generation) % push_rate, 'a', encoding='utf - 8')
    f.write('{}\n'.format(push_rate))
    f.write('1 {}\n'.format(point1))
    f.write('2 {}\n'.format(point2))
    f.write('3 {}\n'.format(point3))
    f.write('4 {}\n'.format(point4))
    f.write('5 {}\n'.format(point5))
    f.write('6 {}\n'.format(point6))
    f.write('7 {}\n'.format(point7))
    f.close()
def Set_ratio(set_a, set_b):

    driver_number = set_b.count(set_a)

    driver_rate  = driver_number
    return driver_rate
def Driver_mutation(X_center,Y_center):
    cell_mutation_ID = cell_dictionary[X_center,Y_center].mutation
    driver = random.sample(cell_mutation_ID,1)
    return driver

def Inter_set(set_a, set_b):
    intersection_set = list(set(set_a).intersection(set(set_b)))
    return intersection_set



def Radiate_simplling(X_center,Y_center):
    XY_center =(X_center,Y_center)
    all_mutation_ID_set=[]
    all_mutation_ID_inset=[]
    all_mutation_ID_sort = []  #remove the Repeated number
    horizon_x11 = []
    horizon_y11 = []
    cancer_num11 =100
    d11 = 5
    for cut in range(0, len(cell_r)):
        x0 = cell_r[cut]
        y0 = cell_c[cut]
        if X_center - d11 <= x0 < X_center + d11 and Y_center - d11 <= y0 < Y_center + d11:
            horizon_x11.append(x0)
            horizon_y11.append(y0)

        single_mutation_ID = cell_dictionary[x0, y0].mutation  # get all the mutation ID in the dictionary
        all_mutation_ID_set += single_mutation_ID  # Collect every mutation.ID in the all_mutation_ID
        ### 11 slice
    for cut11 in range(0, len(horizon_x11)):
        x11 = horizon_x11[cut11]
        y11 = horizon_y11[cut11]
        single_mutation_ID11 = cell_dictionary[x11, y11].mutation  # get all the mutation ID in the dictionary

        all_mutation_ID_inset += single_mutation_ID  # Collect every mutation.ID in the all_mutation_ID1

    return  all_mutation_ID_inset

def Plot_radiation(num,driver,p_distance,x,y,):
    x11 = [0, 1*p_distance,2*p_distance,3*p_distance,4*p_distance,5*p_distance]

    # count_id = Counter(Radiate_simplling(x, y))
    # for i in count_id.keys():
    #     if i != 0 :
    #         driver = i
    #         break
    # print('driver:',driver)


    heatmesh_set= np.zeros((11, 11))
    heatmesh= {}
    global exec
    ("s%sk1" % driver)
    global exec
    ("s%sk2" % driver)
    global exec
    ("s%sk3" % driver)
    global exec
    ("s%sk4" % driver)
    global exec
    ("s%sk6" % driver)
    global exec
    ("s%sk7" % driver)
    global exec
    ("s%sk8" % driver)
    global exec
    ("s%sk9" % driver)

    exec("s%sk1=[]" % driver)
    exec("s%sk2=[]" % driver)
    exec("s%sk3=[]" % driver)
    exec("s%sk4=[]" % driver)
    exec("s%sk6=[]" % driver)
    exec("s%sk7=[]" % driver)
    exec("s%sk8=[]" % driver)
    exec("s%sk9=[]" % driver)

    for i in x11:
        s01 =Set_ratio(driver, Radiate_simplling(x - i, y - i))
        exec("s%sk1.append(s01)" % driver)
        s02 = Set_ratio(driver, Radiate_simplling(x - i, y))
        exec("s%sk2.append(s02)" % driver)
        s03=Set_ratio(driver, Radiate_simplling(x - i, y + i))
        exec("s%sk3.append(s03)" % driver)
        s04=Set_ratio(driver, Radiate_simplling(x , y- i))
        exec("s%sk4.append(s04)" % driver)
        s06 = Set_ratio(driver, Radiate_simplling(x , y+i))
        exec("s%sk6.append(s06)" % driver)
        s07 = Set_ratio(driver, Radiate_simplling(x + i, y - i))
        exec("s%sk7.append(s07)" % driver)
        s08 = Set_ratio(driver, Radiate_simplling(x+i, y ))
        exec("s%sk8.append( s08)" % driver)
        s09 = Set_ratio(driver, Radiate_simplling(x + i, y + i))
        exec("s%sk9.append(s09)" % driver)

        heatmesh[int(5 - i / 10), int(5 - i / 10)] = s01
        heatmesh[(int(5 - i / 10), 5)] = s02
        heatmesh[int(5 - i / 10), int(5 + i / 10)] = s03
        heatmesh[(5), int(5 - i / 10)] = s04
        heatmesh[5, int(5 + i / 10)] = s06
        heatmesh[int(5 + i / 10), int(5 - i / 10)] = s07
        heatmesh[int(5 + i / 10), 5] = s08
        heatmesh[int(5 + i / 10), int(5 + i / 10)] = s09
    # print('driver:', exec("s%sk1" % driver),exec("s%sk9" % driver))
    for i,j in heatmesh.items():
        a, b = i
        heatmesh_set[a, b] = j

    # plt.figure('%s heatmap {}'.format(driver) % num,figsize = [5,5])
    # plt.title('Heatmap ID:{}'.format(driver))
    # plt.imshow(heatmesh_set, cmap= plt.get_cmap('Reds'))
    # print('mmm',heatmesh_set)
    # for i,j in heatmesh.items():
    #     a,b = i
    #     plt.text(b,a,j)
    #
    # colors=['Blue','Yellow','Green']
    # plt.figure('radiation{}'.format(num), figsize=[12, 12])
    #
    #
    # plt.subplot(331)
    # plt.text(35,0.8*max(s1)-num,'Blue:{}'.format(driver) )
    # plt.scatter(x11, s1)
    # plt.plot(x11, s1)
    # plt.subplot(332)
    # plt.title('Sample distance = {}'.format(p_distance))
    # plt.scatter(x11, s2)
    # plt.plot(x11, s2)
    # plt.subplot(333)
    # plt.text(35,0.8*max(s3)-num,'Yellow:{}'.format(driver) )
    # plt.scatter(x11, s3)
    # plt.plot(x11, s3)
    # plt.subplot(334)
    # plt.scatter(x11, s4)
    # plt.plot(x11, s4)
    #
    # plt.subplot(336)
    # plt.scatter(x11, s6)
    # plt.plot(x11, s6)
    # plt.subplot(337)
    # plt.scatter(x11, s7)
    # plt.plot(x11, s7)
    # plt.subplot(338)
    # plt.scatter(x11, s8)
    # plt.plot(x11, s8)
    # plt.subplot(339)
    # plt.scatter(x11, s9)
    # plt.plot(x11, s9)

    return heatmesh_set
def Map_sampling(driver_id):
    heatmap_value = Plot_radiation(i_cut, 10, 10, X_radiation, Y_radiation)
    heatmap_aver = np.append(heatmap_aver1, sum(heatmap_value1.tolist(), []), axis=0)
    return heatmap_aver

def Heatmap_line(driver_ID,A_h):
    line1 = [A_h[5,5],A_h[4,4],A_h[3,3],A_h[2,2],A_h[1,1],A_h[0,0]]
    line2 = [A_h[5,5],A_h[4,5],A_h[3,5],A_h[2,5],A_h[1,5],A_h[0,5]]
    line3 = [A_h[5, 5], A_h[4, 6], A_h[3,7], A_h[2, 8], A_h[1, 9], A_h[0, 10]]
    line4 = [A_h[5, 5], A_h[5, 4], A_h[5, 3], A_h[5, 2], A_h[5, 1], A_h[5, 0]]
    line6 = [A_h[5, 5], A_h[5, 6], A_h[5, 7], A_h[5, 8], A_h[5, 9], A_h[5, 10]]
    line7 = [A_h[5, 5], A_h[6, 4], A_h[7, 3], A_h[8, 2], A_h[9, 1], A_h[10, 0]]
    line8 = [A_h[5, 5], A_h[6, 5], A_h[7, 5], A_h[8, 5], A_h[9, 5], A_h[10, 5]]
    line9 = [A_h[5, 5], A_h[6, 6], A_h[7, 7], A_h[8, 8], A_h[9, 9], A_h[10, 10]]
    p_distance = 10
    x11 = [0, 1*p_distance,2*p_distance,3*p_distance,4*p_distance,5*p_distance]

    plt.figure('radiation', figsize=[12, 12])
    plt.title('Heatmap Line')
    plt.subplot(331)

    plt.scatter(x11, line1)

    plt.plot(x11, line1,label="{}".format(driver_ID))
    plt.legend()
    plt.subplot(332)
    plt.title('Heatmap Line')
    plt.scatter(x11, line2)
    plt.plot(x11, line2)
    plt.subplot(333)

    plt.scatter(x11, line3)
    plt.plot(x11, line3)
    plt.subplot(334)
    plt.scatter(x11, line4)
    plt.plot(x11, line4)

    plt.subplot(336)
    plt.scatter(x11, line6)
    plt.plot(x11, line6)
    plt.subplot(337)
    plt.scatter(x11, line7)
    plt.plot(x11, line7)
    plt.subplot(338)
    plt.scatter(x11, line8)
    plt.plot(x11, line8)
    plt.subplot(339)
    plt.scatter(x11, line9)
    plt.plot(x11, line9)

def Map_average(times,driver_ID,heatmap_aver):
    whole_heatmap=np.zeros((heatmap_gener,121))
    for i in range(0,heatmap_gener):
        whole_heatmap[i,:] = heatmap_aver[0+121*i:121+121*i]
    average_heatmap = np.mean(whole_heatmap,axis=0)
    Average_heatmap = average_heatmap.reshape(11,11)
    # plt.figure(' Average Heatmap %s' %driver_ID, figsize=[6, 6])
    # plt.title('Driver ID : %s, Occur time: %s    {} sampling'.format(heatmap_gener) %(driver_ID,times))
    # plt.imshow(Average_heatmap,cmap= plt.get_cmap('Reds'))
    #
    # for i in range(11):
    #     for j in range(11):
    #         value=Average_heatmap[i,j]
    #         if value != 0:
    #             plt.text(j-0.3, i, value)
    Heatmap_line(driver_ID, Average_heatmap)



class Cancercell ():

    def __init__(self, cp,neighbor_mesh):
        """
               Initialize Cancer Cell
               :param pos: position of cancer cell; tuple
               :param dictionary_of_neighbor_mesh: used to tell the cell the positions of its neighbors

        """
        self.ip = 0
        self.cp = cp  # cp means central position
        self.die_divide_times = die_divide_times  # define the times of cell divide
        self.neighbor_pos =neighbor_mesh[cp]  # define the neighbor central point
        self.ID = 1  # the cell ID type, 1 main back_cell
        self.mutation = list(range(background_gene))  # mutation data add the back gene inform
        self.mu_times = 0  # the initial divide times of mutation divide
        self.pu_times = Max_generation



        global times

        times = 0

    def empty_neighbor(self, agent):
        """
                Search for empty positions in More neighborhood. If there is more than one free position,
                randomly select one and return it
                :param agent: dictionary of agents, key=position, value = cell.ID; dict
                :return: Randomly selected empty position, or None if no empty positions
        """
        diagnal_empty = []
        square_empty = []
        count = 0

        empty_neighbor_list = [neighbor for neighbor in self.neighbor_pos if neighbor not in agent ]
        # print('1',empty_neighbor_list)
        # for i in self.neighbor_pos:
        #     if i in all_mutation_del:
        #         print('zzzzzzzzzzzzzz')
        #         empty_neighbor_list.append(i)
        #
        # print('2', empty_neighbor_list)

        if agent[self.cp].mutation == [-1]:
            count += 1

        for empty in empty_neighbor_list:
            Er, Ec = empty
            r, c = self.cp

            if r-Er == 0 or c-Ec == 0 :
                square_empty.append(empty)
            else:
                diagnal_empty.append(empty)

        if empty_neighbor_list :
            # diagnal_pushing_weight = diagnal_empty / diagnal_empty = 1 / 2
            diagnal_weight = len(diagnal_empty) / (8*(len(diagnal_empty) +  len(square_empty)))
            diagnal = binomial(1, diagnal_weight)

            if diagnal_empty and diagnal == 1:
                empty_pos = random_choice(diagnal_empty)
            elif square_empty and diagnal != 1:
                empty_pos = random_choice(square_empty)
            elif not diagnal_empty:
                empty_pos = random_choice(square_empty)
            elif not square_empty:
                empty_pos = random_choice(diagnal_empty)


            return empty_pos

        else:
            return None

    def pushing_pos(self, agent):
        pushing_pos = [cp for cp in self.neighbor_pos ]
        empty_neighbor_list = [cp for cp in self.neighbor_pos if cp not in agent ]
        if empty_neighbor_list is None:
            return pushing_pos

    def act(self, agent, neighbor_mesh):
        """
                Cell carries out its actions, which are division and death. Cell will divide if it is lucky and
                there is an empty position in its neighborhood. Cell dies either spontaneously or if it exceeds its
                maximum number of divisions.

                :param agent: dictionary of agents, key=position, value = cell.ID; dict
                :return: None
        """
        divide = birth_r()

        if divide == 1:
            poisson1 = 1
            poisson2 = 1
            # print('poisson = ',poisson)
            empty_pos = self.empty_neighbor(agent)      # input empty position
            pushing = push_r()      # input the push rate
            if empty_pos is not None :   # Growth on the empty position
                # print('division to empty space',self.cp,empty_pos)
                # Creat new daughter cell and it to the cell dictionary
                daughter_cell = Cancercell(empty_pos,neighbor_mesh)

                # define the daughter mutation_ID's background same as parent add the new mutation_ID by poisson
                agent[empty_pos] = daughter_cell

                all_mutation.append(max(self.mutation))

                daughter_cell.mutation = agent[self.cp].mutation + list(
                    range(max(all_mutation + agent[self.cp].mutation) + 1,
                          max(all_mutation + agent[self.cp].mutation) + 1 + poisson1))
                agent[self.cp].mutation = agent[self.cp].mutation + list(
                    range(max(all_mutation + daughter_cell.mutation) + 1,
                          max(all_mutation + daughter_cell.mutation) + 1 + poisson2))
                all_mutation.append(max(daughter_cell.mutation))
                all_mutation.append(max(agent[self.cp].mutation))


                # define the cell ip in the tissue
                all_ip.append(agent[self.cp].ip)
                daughter_cell.ip = max(all_ip) + 1
                all_ip.append(daughter_cell.ip)
                agent[self.cp].ip = max(all_ip) + 1
                all_ip.append( agent[self.cp].ip)


                if mutation_type2 in self.mutation:
                    daughter_cell.ID = 2

                elif mutation_type3 in self.mutation:
                    daughter_cell.ID = 3
                    agent[self.cp].ID = 3
                elif mutation_type4 in self.mutation:
                    daughter_cell.ID = 4
                    agent[self.cp].ID = 4
                elif mutation_type5 in self.mutation:
                    daughter_cell.ID = 5
                    agent[self.cp].ID = 5
                #heatmap define for push rate
                # cancer_heatmap[self.cp]=len(self.mutation)
                # cancer_heatmap[empty_pos] = len(daughter_cell.mutation)

                cancer_matrix[self.cp] = agent[self.cp].ip


                cancer_matrix[empty_pos] = daughter_cell.ip

            if empty_pos is None and pushing == 1:
                # print('Here to pushing')
                # Count the number of living cancer around the cell

                new_r, new_c = (0, 1)
                # random choice the directions to pushing
                set_local = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
                new_r, new_c = random_choice(set_local)
                #pushing weight percent (without push_weight in this algrithm)
                # diagnal = binomial(1, diagnal_pushing_weight)
                # if diagnal == 1:
                #     new_r, new_c =random_choice(diagnal_pushing)
                # else:
                #     new_r, new_c = random_choice(square_pushing)

                # center position and daughter cell
                cp_r, cp_c = self.cp
                newcell_pos = (cp_r + new_r, cp_c + new_c)
                newcell2_pos = (cp_r + new_r + new_r, cp_c + new_c + new_c)
                # print('Pushing cp',self.cp, 'newcell_pos', newcell_pos)
                #push_direction,in total 9 directions, 0-8 describe the 3*3 matrix，center_position = 4 point

                if new_r == -1 and new_c == -1:  # 0 point
                    push_matrix = []
                    if cp_r >= cp_c:
                        for i in range(cp_c):
                            push_matrix = np.append(push_matrix, cancer_matrix[cp_r - i, cp_c - i])
                            if 0 in push_matrix:
                                break
                        push_matrix = np.delete(push_matrix, [len(push_matrix) -1])
                        push_matrix = np.insert(push_matrix, 0, 1)
                        daughter_cell3 = Cancercell((cp_r - (len(push_matrix)) + 1, cp_c - (len(push_matrix)) + 1),
                                                   neighbor_mesh)
                        agent[cp_r - (len(push_matrix)) + 1, cp_c - (len(push_matrix)) + 1] = daughter_cell3
                        newcell2_pos=(cp_r - (len(push_matrix)) + 1, cp_c - (len(push_matrix)) + 1)

                        for i in range(len(push_matrix)):
                            cancer_matrix[cp_r - i, cp_c - i] = push_matrix[i]
                        for i in range(len(push_matrix)-1):
                            agent[cp_r - len(push_matrix) + 1 + i, cp_c - len(push_matrix) + 1 + i].mutation = agent[cp_r - len(push_matrix) + 2 + i, cp_c - len(push_matrix) + 2 + i].mutation

                            # move the whole dictionary
                            agent[cp_r - len(push_matrix) + 1 + i, cp_c - len(push_matrix) + 1 + i].ip = agent[
                                cp_r - len(push_matrix) + 2 + i, cp_c - len(push_matrix) + 2 + i].ip

                    else:
                        for i in range(cp_r):
                            push_matrix = np.append(push_matrix, cancer_matrix[cp_r - i, cp_c - i])
                            if 0 in push_matrix:
                                break
                        push_matrix = np.delete(push_matrix, [len(push_matrix)-1])
                        push_matrix = np.insert(push_matrix, 0, 1)
                        daughter_cell3 = Cancercell((cp_r - (len(push_matrix)) + 1, cp_c - (len(push_matrix)) + 1),
                                                   neighbor_mesh)
                        agent[cp_r - (len(push_matrix)) + 1, cp_c - (len(push_matrix)) + 1] = daughter_cell3
                        newcell2_pos = (cp_r - (len(push_matrix)) + 1, cp_c - (len(push_matrix)) + 1)
                        for i in range(len(push_matrix)):
                            cancer_matrix[cp_r - i , cp_c - i] = push_matrix[i]
                        for i in range(len(push_matrix)-1):
                            agent[cp_r - (len(push_matrix)) + 1 + i, cp_c - (len(push_matrix)) +1+ i].mutation = agent[
                                cp_r - (len(push_matrix)) + 2 + i, cp_c - (len(push_matrix)) + 2 + i].mutation
                            agent[cp_r - (len(push_matrix)) + 1 + i, cp_c - (len(push_matrix)) + 1 + i].ip = agent[cp_r - (len(push_matrix)) + 2 + i, cp_c - (len(push_matrix)) + 2 + i].ip

                elif new_r == -1 and new_c == 0:  # 1 point
                    push_matrix = []
                    for i in range(cp_r):
                        push_matrix = np.append(push_matrix, cancer_matrix[cp_r-i, cp_c:(cp_c+1)])
                        if 0 in push_matrix:
                            break

                    push_matrix = np.delete(push_matrix, [len(push_matrix)-1])
                    push_matrix = np.insert(push_matrix,0, 1)
                    agent[cp_r - (len(push_matrix) - 1), cp_c] = Cancercell((cp_r - (len(push_matrix) - 1), cp_c), neighbor_mesh )
                    newcell2_pos = (cp_r - (len(push_matrix) - 1), cp_c)
                    for i in range(len(push_matrix)):
                        cancer_matrix[cp_r - i, cp_c] = push_matrix[i]
                    for i in range(len(push_matrix)-2):
                        agent[cp_r - (len(push_matrix)) + 1 + i, cp_c].mutation = agent[cp_r - (len(push_matrix)) + 2 + i, cp_c].mutation
                        agent[cp_r - (len(push_matrix)) + 1 + i, cp_c].ip = agent[
                            cp_r - (len(push_matrix)) + 2 + i, cp_c].ip

                elif new_r == -1 and new_c == 1:  # 2 point
                    push_matrix = []
                    if (cp_r+1)  >=  (Max_COL - cp_c):
                        for i in range(Max_COL - cp_c):
                            push_matrix = np.append(push_matrix, cancer_matrix[cp_r - i, cp_c + i])
                            if 0 in push_matrix:
                                break
                        push_matrix = np.delete(push_matrix, [len(push_matrix)-1])

                        push_matrix = np.insert(push_matrix, 0, 1)
                        agent[cp_r - (len(push_matrix) - 1), cp_c + (len(push_matrix)) - 1] = Cancercell(
                            (cp_r - (len(push_matrix) - 1), cp_c + (len(push_matrix)) - 1), neighbor_mesh)
                        newcell2_pos = (cp_r - (len(push_matrix) - 1), cp_c + (len(push_matrix)) - 1)
                        for i in range(len(push_matrix)):
                            cancer_matrix[cp_r - i, cp_c + i] = push_matrix[i]
                        for i in range(len(push_matrix)-2):
                            agent[cp_r - (len(push_matrix) - 1-i), cp_c + (len(push_matrix) - 1-i)].mutation = agent[cp_r - (len(push_matrix) - 2-i), cp_c + (len(push_matrix) - 2-i)].mutation
                            # if i ==(len(push_matrix)-3):
                                # print('www',agent[cp_r - (len(push_matrix) - 1 - i), cp_c + (len(push_matrix) - 1 - i)].mutation ,'\n',cp_r - (len(push_matrix) -2 - i), cp_c + (len(push_matrix) -2  - i), newcell_pos,agent[cp_r - (len(push_matrix)-2  - i), cp_c + (len(push_matrix) - 2 - i)].mutation)
                            agent[cp_r - (len(push_matrix) - 1 - i), cp_c + (len(push_matrix) - 1 - i)].ip = agent[cp_r - (len(push_matrix) - 2 - i), cp_c + (len(push_matrix) - 2 - i)].ip
                    elif (cp_r+1)  <  (Max_COL - cp_c):
                        for i in range(cp_r+1):
                            push_matrix = np.append(push_matrix, cancer_matrix[cp_r - i, cp_c + i])
                            if 0 in push_matrix:
                                break
                        push_matrix = np.delete(push_matrix, [len(push_matrix)-1])
                        push_matrix = np.insert(push_matrix, 0, 1)

                        agent[cp_r - (len(push_matrix) - 1), cp_c + (len(push_matrix)) - 1] = Cancercell(
                            (cp_r - (len(push_matrix) - 1), cp_c + (len(push_matrix)) - 1), neighbor_mesh)
                        newcell2_pos = (cp_r - (len(push_matrix) - 1), cp_c + (len(push_matrix)) - 1)
                        for i in range(len(push_matrix)):
                            cancer_matrix[cp_r - i, cp_c+ i ] = push_matrix[i]
                        for i in range(len(push_matrix)-2):
                            agent[cp_r - (len(push_matrix) - 1-i), cp_c + (len(push_matrix) - 1-i)].mutation = agent[cp_r - (len(push_matrix) - 2-i), cp_c + (len(push_matrix) - 2-i)].mutation
                            agent[cp_r - (len(push_matrix) - 1 - i), cp_c + (len(push_matrix) - 1 - i)].ip = agent[cp_r - (len(push_matrix) - 2 - i), cp_c + (len(push_matrix) - 2 - i)].ip


                elif new_r == 0 and new_c == -1:  # 3 point
                    push_matrix = []
                    for i in range(cp_c):
                        push_matrix = np.append(push_matrix, cancer_matrix[cp_r:(cp_r+1), cp_c-i])
                        if 0 in push_matrix:
                            break

                    push_matrix = np.delete(push_matrix, [len(push_matrix)-1])
                    push_matrix = np.insert(push_matrix, 0, 1)
                    agent[cp_r, cp_c - (len(push_matrix) - 1)] = Cancercell((cp_r, cp_c - (len(push_matrix) - 1)),
                                                                            neighbor_mesh)
                    newcell2_pos = (cp_r, cp_c - (len(push_matrix) - 1))
                    for i in range(len(push_matrix)):
                        cancer_matrix[cp_r, cp_c - i] = push_matrix[i]
                    for i in range(len(push_matrix)-2):
                        agent[cp_r , cp_c - (len(push_matrix) - 1 - i)].mutation = agent[
                            cp_r, cp_c - (len(push_matrix) - 2 - i)].mutation
                        agent[cp_r, cp_c - (len(push_matrix) - 1 - i)].ip = agent[
                            cp_r, cp_c - (len(push_matrix) - 2 - i)].ip
                elif new_r == 0 and new_c == 1:  # 5 point
                    push_matrix = []
                    for i in range(Max_COL- cp_c):
                        push_matrix = np.append(push_matrix, cancer_matrix[cp_r:(cp_r + 1), cp_c+i])
                        if 0 in push_matrix:
                            break

                    push_matrix = np.delete(push_matrix, [len(push_matrix) - 1])
                    push_matrix = np.insert(push_matrix, 0, 1)
                    agent[cp_r, cp_c + (len(push_matrix) - 1)] = Cancercell((cp_r, cp_c + (len(push_matrix) - 1)),
                                                                            neighbor_mesh)
                    newcell2_pos = (cp_r, cp_c + (len(push_matrix) - 1))
                    for i in range(len(push_matrix)):
                        cancer_matrix[cp_r, cp_c + i] = push_matrix[i]
                    for i in range(len(push_matrix)-2):
                        agent[cp_r, cp_c + (len(push_matrix) - 1 - i)].mutation = agent[
                            cp_r, cp_c + (len(push_matrix) - 2 - i)].mutation
                        agent[cp_r, cp_c + (len(push_matrix) - 1 - i)].ip = agent[
                            cp_r, cp_c + (len(push_matrix) - 2 - i)].ip

                elif new_r == 1 and new_c == -1:    # 6 point
                    push_matrix = []
                    if cp_c+1 >= (Max_ROW - cp_r):
                        for i in range(Max_ROW - cp_r):
                            push_matrix = np.append(push_matrix, cancer_matrix[cp_r + i, cp_c - i])
                            if 0 in push_matrix:
                                break
                        push_matrix = np.delete(push_matrix, [len(push_matrix)-1])
                        push_matrix = np.insert(push_matrix, 0, 1)
                        agent[cp_r + (len(push_matrix) - 1), cp_c - (len(push_matrix)) + 1] = Cancercell(
                            (cp_r + (len(push_matrix) - 1), cp_c - (len(push_matrix)) + 1), neighbor_mesh)
                        newcell2_pos = (cp_r + (len(push_matrix) - 1), cp_c - (len(push_matrix)) + 1)
                        for i in range(len(push_matrix)):
                            cancer_matrix[cp_r + i, cp_c - i] = push_matrix[i]
                        for i in range(len(push_matrix)-2):
                            agent[cp_r + (len(push_matrix) - 1 - i), cp_c - (len(push_matrix) - 1 - i)].mutation = agent[
                                cp_r + (len(push_matrix) - 2 - i), cp_c - (len(push_matrix) - 2 - i)].mutation
                            agent[cp_r + (len(push_matrix) - 1 - i), cp_c - (len(push_matrix) - 1 - i)].ip = agent[ cp_r + (len(push_matrix) - 2 - i), cp_c - (len(push_matrix) - 2 - i)].ip
                    elif cp_c < (Max_ROW - cp_r):
                        for i in range(cp_c):
                            push_matrix = np.append(push_matrix, cancer_matrix[cp_r + i, cp_c - i])
                            if 0 in push_matrix:
                                break
                        push_matrix = np.delete(push_matrix, [len(push_matrix)-1])

                        push_matrix = np.insert(push_matrix, 0, 1)
                        agent[cp_r + (len(push_matrix) - 1), cp_c - (len(push_matrix)) + 1] = Cancercell(
                            (cp_r + (len(push_matrix) - 1), cp_c - (len(push_matrix)) + 1), neighbor_mesh)
                        newcell2_pos = (cp_r + (len(push_matrix) - 1), cp_c - (len(push_matrix)) + 1)
                        for i in range(len(push_matrix)):
                            cancer_matrix[cp_r + i, cp_c - i] = push_matrix[i]
                        for i in range(len(push_matrix)-2):
                            agent[cp_r + (len(push_matrix) - 1 - i), cp_c - (len(push_matrix) - 1 - i)].mutation = agent[
                                cp_r + (len(push_matrix) - 2 - i), cp_c - (len(push_matrix) - 2 - i)].mutation
                            agent[cp_r + (len(push_matrix) - 1 - i), cp_c - (len(push_matrix) - 1 - i)].ip = agent[
                                cp_r + (len(push_matrix) - 2 - i), cp_c - (len(push_matrix) - 2 - i)].ip

                elif new_r == 1 and new_c == 0:  # 7 point
                    push_matrix = []
                    for i in range(Max_ROW - cp_r):
                        push_matrix = np.append(push_matrix, cancer_matrix[cp_r + i, cp_c:(cp_c + 1) ])
                        if 0 in push_matrix:
                            break

                    push_matrix = np.delete(push_matrix, [len(push_matrix) - 1])
                    push_matrix = np.insert(push_matrix, 0, 1)
                    agent[cp_r + (len(push_matrix) - 1), cp_c] = Cancercell((cp_r + (len(push_matrix) - 1), cp_c),
                                                                            neighbor_mesh)
                    newcell2_pos = (cp_r + (len(push_matrix) - 1), cp_c)
                    for i in range(len(push_matrix)):
                        cancer_matrix[cp_r + i, cp_c] = push_matrix[i]
                    for i in range(len(push_matrix)-2):
                        agent[cp_r + (len(push_matrix) - 1 - i), cp_c].mutation = agent[cp_r + (len(push_matrix) - 2 - i), cp_c ].mutation
                        agent[cp_r + (len(push_matrix) - 1 - i), cp_c].ip = agent[
                            cp_r + (len(push_matrix) - 2 - i), cp_c].ip

                elif new_r == 1 and new_c == 1:  # 8 point
                    push_matrix = []
                    if (Max_ROW - cp_r) >= (Max_COL - cp_c):
                        for i in range(Max_COL - cp_c):
                            push_matrix = np.append(push_matrix, cancer_matrix[cp_r + i, cp_c + i])
                            if 0 in push_matrix:
                                break
                        push_matrix = np.delete(push_matrix, [len(push_matrix)-1])
                        push_matrix = np.insert(push_matrix, 0, 1)
                        agent[cp_r + (len(push_matrix) - 1), cp_c + (len(push_matrix)) - 1] = Cancercell(
                            (cp_r + (len(push_matrix) - 1), cp_c + (len(push_matrix)) - 1), neighbor_mesh)

                        newcell2_pos = (cp_r + (len(push_matrix) - 1), cp_c + (len(push_matrix)) - 1)
                        for i in range(len(push_matrix)):
                            cancer_matrix[cp_r + i, cp_c + i] = push_matrix[i]
                        for i in range(len(push_matrix)-2):
                            agent[cp_r + (len(push_matrix) - 1 - i), cp_c + (len(push_matrix) - 1 - i)].mutation = agent[
                                cp_r + (len(push_matrix) - 2 - i), cp_c + (len(push_matrix) - 2 - i)].mutation
                            agent[cp_r + (len(push_matrix) - 1 - i), cp_c + (len(push_matrix) - 1 - i)].ip = agent[
                                cp_r + (len(push_matrix) - 2 - i), cp_c + (len(push_matrix) - 2 - i)].ip

                    else:
                        for i in range(Max_ROW - cp_r):
                            push_matrix = np.append(push_matrix, cancer_matrix[cp_r + i, cp_c + i])
                            if 0 in push_matrix:
                                break
                        push_matrix = np.delete(push_matrix, [len(push_matrix)-1])
                        push_matrix = np.insert(push_matrix, 0, 1)
                        agent[cp_r + (len(push_matrix) - 1), cp_c + (len(push_matrix)) - 1] = Cancercell(
                            (cp_r + (len(push_matrix) - 1), cp_c + (len(push_matrix)) - 1), neighbor_mesh)
                        newcell2_pos = (cp_r + (len(push_matrix) - 1), cp_c + (len(push_matrix)) - 1)
                        for i in range(len(push_matrix)):
                            cancer_matrix[cp_r + i, cp_c + i] = push_matrix[i]
                        for i in range(len(push_matrix)-2):
                            agent[cp_r + (len(push_matrix) - 1 - i), cp_c + (len(push_matrix) - 1 - i)].mutation = agent[cp_r + (len(push_matrix) - 2 - i), cp_c + (len(push_matrix) - 2 - i)].mutation
                            agent[cp_r + (len(push_matrix) - 1 - i), cp_c + (len(push_matrix) - 1 - i)].ip = agent[
                                cp_r + (len(push_matrix) - 2 - i), cp_c + (len(push_matrix) - 2 - i)].ip

                list_newcell.append(newcell_pos)        # record every newcell
                list_newcell2.append(newcell2_pos)


                # define the mutation ID to the new cell
                all_mutation.append(max(self.mutation))
                agent[newcell_pos].mutation = agent[self.cp].mutation + list(
                    range(max(all_mutation + agent[self.cp].mutation) + 1,
                          max(all_mutation + agent[self.cp].mutation) + 1 + poisson1))
                agent[self.cp].mutation = agent[self.cp].mutation + list(
                    range(max(all_mutation + agent[newcell_pos].mutation) + 1,
                          max(all_mutation + agent[newcell_pos].mutation) + 1 + poisson2))
                all_mutation.append(max(agent[newcell_pos].mutation))
                all_mutation.append(max(agent[self.cp].mutation))

                # define the new ip to the new cell
                all_ip.append(agent[self.cp].ip)
                agent[newcell_pos].ip = max(all_ip) + 1
                all_ip.append(agent[newcell_pos].ip)
                agent[self.cp].ip = max(all_ip) + 1

                # f = open('ush-parents.csv', 'a', encoding='utf - 8')
                # f.write('s{}'.format(max(all_ip)))
                # f.write(' {}\n'.format(agent[newcell_pos].ip))
                # f.write(' {}'.format(max(all_ip)))
                # f.write(' {}\n'.format(agent[self.cp].ip))
                # f.close()
                all_ip.append(agent[self.cp].ip)



                # cancer_heatmap[newcell_pos] = len(agent[newcell_pos].mutation)
                # cancer_heatmap[newcell2_pos] = len(agent[newcell2_pos].mutation)
                # cancer_heatmap[self.cp]= len( agent[self.cp].mutation)

                if mutation_type1 in self.mutation:
                    agent[newcell_pos].ID = 1
                    agent[self.cp].ID =1
                if mutation_type2 in self.mutation:
                    agent[newcell_pos].ID = 2
                    agent[self.cp].ID =2
                elif mutation_type3 in self.mutation:
                    agent[newcell_pos].ID = 3
                    agent[self.cp].ID = 3
                elif mutation_type4 in self.mutation:
                    agent[newcell_pos].ID = 4
                    agent[self.cp].ID = 4
                elif mutation_type5 in self.mutation:
                    agent[newcell_pos].ID = 5
                    agent[self.cp].ID = 5



            # print('self',self.cp)
            '''the cancer cell death process'''
            # the cell death algorithm (spontaneous death) and (divide death)
            self.die_divide_times -= 1   # The life span of a cell divided is reduced by one
            spontaneous_death = death_r()
            # (add divide death code: 'or self.die_divide_times <= 0')
            if spontaneous_death == 1:
                cancer_matrix[self.cp] = 0
                agent[self.cp].mutation = [-1]
                all_mutation_del.append(self.cp)
                # agent[self.cp].ip = [0]

            # print('empty', all_mutation_del)


if __name__ == "__main__":
    mutation_ID = background_gene-1
    cell_replication = Max_generation       # the number of cell divide generation
    cell_matrix = np.zeros(mesh_shape)      # create a whole growth mesh
    cancer_grid = cancer_mesh(Max_ROW, Max_COL)     # define the cancer tissue growth mesh space
    central_r = int(round(Max_ROW/2))       # the center of row
    central_c = int(round(Max_COL/2))       # the center of col
    central_pos = (central_r, central_c)    # the center position
    initial_cell = Cancercell(central_pos, cancer_grid)     # define the dictionary of cancer
    cell_dictionary = {central_pos: initial_cell}       # define the whole mesh dictionary
    mutation_dictionary = {central_pos: mutation_ID}    # define the mutation ID dictionary
    mutations_dictionary = {mutation_ID: central_pos}     # define the whole mutation ID dictionary
    tissue_dictionary = {}      # Create the empty tissue dictionary
    gener_num_dict = {}
    # Create the update list dictionary
    list1 = {}
    list2 = {}
    list3 = {}

    ###
    mutation_num1 = 0
    mutation_num2 = 0
    mutation_num3 = 0
    mutation_num4 = 0
    mutation_num5 = 0
    mutation_num6 = 0
    mutation_num7 = 0
    all_cp =[]
    driver_time = []
    driver_time1 = []
    driver_time2 = []
    driver_time3 = []
    driver_time4 = []
    driver_time5 = []
    driver_time6 = []
    sum_cancer =[]
    for rep in range(cell_replication):   # replicate cell in the generation
        cell_list = list(cell_dictionary.items())
        for ID in range(len(cell_list)):
            tissue_dictionary.update({ID: list(cell_dictionary.items())[ID]})
        grow_list = list(mutations_dictionary.keys())
        # get the total cell in list

        # print(grow_list)
        muller_mutation =[]
        shuffle(grow_list)
        for key in  grow_list:  # Iterate through cell_dictionary to find the center point
            cp = mutations_dictionary[key]
            cell = cell_dictionary[cp]
            cell.cp = cp
            c_r, c_c = cell.cp
            cell_dictionary[cell.cp].mutation = cell.mutation   # Assign the mutation ID to each center point
            cancer_matrix[cell.cp] = cell.ip  # Assign the ID to each center point

            muller_mutation += cell.mutation

            if mutation_type1 in cell.mutation:
                driver_time1.append(rep)
            if mutation_type2 in cell.mutation:
                driver_time2.append(rep)
            if mutation_type3 in cell.mutation:
                driver_time3.append(rep)
            if mutation_type4 in cell.mutation:
                driver_time4.append(rep)
            if mutation_type5 in cell.mutation:
                driver_time5.append(rep)
            if mutation_type6 in cell.mutation:
                driver_time6.append(rep)

        # shuffle(cell_list)      # random growth order
        muller_result = Counter(muller_mutation)
        muller_size = []  # empty list: record the mutation ID in the VAF
        muller_number = []


        for i in muller_result:
            muller_size.append(i)
        list.sort(muller_size)
        for j in muller_size:
            muller_number.append(muller_result[j])

        f = open('data_muller_number%s+{}.csv'.format(Max_generation) % push_rate, 'a', encoding='utf - 8')
        f.write('{}'.format(rep))
        f.write(',{}\n'.format(muller_number))
        f.close()

        for key in grow_list:  # Iterate through the list to run act
            cp = mutations_dictionary[key]
            cell = cell_dictionary[cp]
            cell.cp = cp
            cancer_matrix[cell.cp] = cell.ip

            if cell.mutation != [-1]:
                cell.act(cell_dictionary, cancer_grid)

                # show the list table
                for key in cell_dictionary.keys():
                    mutation_dictionary.update({key: cell_dictionary[key].ip})

                mutations_dictionary = dict(zip(mutation_dictionary.values(), mutation_dictionary.keys()))
        sum_cancer.append(np.sum(cancer_matrix != 0))
        gener_num_dict[rep] = np.sum(cancer_matrix != 0)
    f = open('cancer_number.csv', 'a', encoding='utf - 8')
    f.write('{}'.format(push_rate))

    f.write(',{}\n'.format(sum_cancer))
    f.close()

    cell_r = []
    cell_c = []
    cell_r1 = []
    cell_c1 = []
    # print("long list",len(list(cell_dictionary.items())))

    for key, cell in list(cell_dictionary.items()):   # Iterate through cell_dictionary to find the center point
        all_cp.append(key)
        cell.cp = key
        c_r, c_c = cell.cp
        # print('cp', cp, cell.mutation)

        f = open('data_muller_mutation%s+{}.csv'.format(Max_generation) % push_rate, 'a', encoding='utf - 8')

        f.write('{}\n'.format(cell.mutation))
        f.close()
        cell_dictionary[cell.cp].mutation = cell.mutation
        list_mutation.append(cell.cp)

        # cell_dictionary[cell.cp].mutation = cell.mutation
        list_mutation.append(cell.cp)
        cancer_heatmap[cell.cp] = len(cell_dictionary[cell.cp].mutation)
        if 0 in cell.mutation :
            cell_matrix[cell.cp] = 1
        if mutation_type2 in cell.mutation:
            cell_matrix[cell.cp] = 2
        if mutation_type3 in cell.mutation:
            cell_matrix[cell.cp] = 3
        if mutation_type4 in cell.mutation:
            cell_matrix[cell.cp] = 4
        if mutation_type5 in cell.mutation:
            cell_matrix[cell.cp] = 5

        if cell.mutation == [0]:
            cell_matrix[cell.cp] = 0

        cell_r.append(c_r)
        cell_c.append(c_c)
        if mutation_type2 in cell.mutation:
            mutation_num2 += 1
        if mutation_type3 in cell.mutation:
            mutation_num3 += 1
        if mutation_type4 in cell.mutation:
            mutation_num4 += 1
        if mutation_type5 in cell.mutation:
            mutation_num5 += 1

    all_mutation.append(max(cell.mutation))

    mutation_num1 = np.sum(cancer_matrix == 1)
    cancer_num = np.sum(cell_matrix != 0)
    if cancer_num != 0:
        cut_point_r = []
        cut_point_c = []
        for i in range(Max_ROW):
            for j in range(Max_COL):
                if cancer_matrix[i, j] != 0:
                    cut_point_r.append(i)
                    cut_point_c.append(j)

        cut_point1 = min(cut_point_r)
        cut_point2 = max(cut_point_r)
        cut_point3 = min(cut_point_c)
        cut_point4 = max(cut_point_c)




    XY_range1 =[]
    XY_range2 = []
    XY_range3 = []
    XY_range4 = []
    XY_range5 = []
    XY_range6 = []

    all_VAF_f11 =[]
    all_VAF_f12 = []
    all_VAF_f13 = []
    all_VAF_f14 = []
    all_VAF_f15 = []
    all_VAF_f16 = []
    all_cell_burden = []
    all_mutation_ID = []
    all_VAF_number11 = []
    all_VAF_number12 = []
    all_VAF_number13 = []
    all_VAF_number14 = []
    all_VAF_number15 = []
    all_VAF_number16 = []
    all_VAF_cumulative11 = []
    all_VAF_cumulative12 = []
    all_VAF_cumulative13 = []
    all_VAF_cumulative14 = []
    all_VAF_cumulative15 = []
    all_VAF_cumulative16 = []
    all_mutation_dict1 = {}
    all_mutation_dict2 = {}
    all_mutation_dict3 = {}
    all_mutation_dict4 = {}
    all_mutation_dict5 = {}
    all_mutation_dict6 = {}


    for cut in range(0, len(cell_r)):
        x0 = cell_r[cut]
        y0 = cell_c[cut]
        mutation_ID = cell_dictionary[x0, y0].mutation  # get all the mutation ID in the dictionary
        all_mutation_ID += mutation_ID  # Collect every mutation.ID in the all_mutation_ID
        all_cell_burden.append(len(mutation_ID))
    # counter the mutation burden in the whole tissue
    burden_result = Counter(all_cell_burden)
    burden_size = []  # empty list: record the mutation ID in the VAF
    burden_number = []
    hist_burden_size = []  # empty list: record the mutation ID in the VAF
    hist_burden_number = []

    for i in burden_result:
        burden_size .append(i)
    list.sort(burden_size, reverse=1)
    for j in burden_size:
        burden_number.append(burden_result[j])

    set_burden_size = [burden_size[i:i + 10] for i in range(0, len(burden_size), 10)]
    set_burden_number = [burden_number[i:i + 10] for i in range(0, len(burden_number), 10)]
    for i in range(len(set_burden_size)):
        hist_burden_size.append(np.mean(set_burden_size[i]))
        hist_burden_number.append(sum(set_burden_number[i]))

    plt.figure('burden distribution', figsize=[5, 5])
    rv = st.poisson(Max_generation * Poisson_lambda)
    x_hist = []
    y_hist = []
    for i in range(len(burden_size)):
        if burden_size[i] < (max(burden_size) - 50):
            x_hist.append(burden_size[i])
            y_hist.append(burden_number[i])


    plt.plot(burden_size, sum(burden_number) * rv.pmf(burden_size), ls='dashed',
             lw=2, c='r', label='Poisson\n$(\lambda={})$\n Generation:%s'.format(Poisson_lambda) % Max_generation)
    plt.bar(burden_size, burden_number, width=1,)
    plt.legend(loc='best')
    plt.xlabel('Burden size')
    plt.ylabel("Number of Cell")


    # plt.figure('hist_burden_number distrabution', figsize=[5, 5])
    # rv = st.poisson(Max_generation * Poisson_lambda)
    # x_hist = []
    # y_hist = []
    # for i in range(len(hist_burden_size)):
    #     if hist_burden_size[i] < (max(hist_burden_size) - 50):
    #         x_hist.append(hist_burden_size[i])
    #         y_hist.append(hist_burden_number[i])
    #
    # Fit_line(x_hist, y_hist)
    #
    # plt.bar(hist_burden_size, hist_burden_number,width=5,label='Poisson\n$(\lambda={})$\n Generation:%s'.format(Poisson_lambda) % Max_generation)
    # plt.xlabel('Burden size')
    # plt.legend(loc='best')
    # plt.ylabel("Number of Cell")


    # counter the mutation in the clip area
    result = Counter(all_mutation_ID)  # Count frequency of each mutation ID
    count_mu = []  # empty list: record the mutation ID in the VAF
    count_times = []  # empty list: record the frequency of mutation ID in the VAF
    '''add the filter algrithm to ignore and analysis the mutation frequency under some value'''
    for i in result:
        count_mu.append(i)
    for j in count_mu:
        count_times.append(result[j])
    if Heatmap_remove_frequency == True:
        remove_mutation=[]
        for k in range(len(count_times)):
            if count_times[k] <= remove_frequency:
                remove_mutation.append(count_mu[k])

        print('Finis simulation')
        for  key, cell in list(cell_dictionary.items()):
            cell.cp = key
            for i in  remove_mutation:
                if i in cell_dictionary[cell.cp].mutation:
                    cell_dictionary[cell.cp].mutation.remove(i)
            filter_heatmap[cell.cp] = len(cell_dictionary[cell.cp].mutation)



    # counter the mutation proberbility in the clip area
    VAF = list(map(lambda n: n / (2 * cancer_num),
                   count_times))  # Object function change to list, and calculate each element of count_times
    result_VAF = (Counter(VAF))  # Count each element in VAF, feedback the numbers
    VAF_mu = []  # empty list: record the mutation ID in the VAF
    VAF_times = []  # empty list: record the each ID frequency in the VAF
    VAF_zoom = {}
    for i1 in result_VAF:
        VAF_mu.append(i1)
    list.sort(VAF_mu, reverse=1)
    for j1 in VAF_mu:
        VAF_times.append(result_VAF[j1])
        VAF_zoom[result_VAF[j1]] =j1

    VAF_cumulative = []

    VAF_number = list(VAF_times)
    for f in range(1, len(VAF_number) + 1):
        VAF_cumulative.append(np.sum(VAF_number[0:f]))
    VAF_f = list(map(lambda c: 1 / c, VAF_mu))

    yvals = list(map(lambda c: 10 * c-20, VAF_f))


    ##### filter histogram

     
     
     
     

    if Heatmap_gap_boxplot == True:
        for i_ra in range(1,sampling_size+1):

            # cut the tissue into 5 piece
            # cancer_num17 = np.sum(cell_matrix[X_center-d1:X_center+d1, 15:255] != 0)
            # cancer_num18 = np.sum(cell_matrix[X_center-d1:175, 15:255] != 0)
            # cancer_num19 = np.sum(cell_matrix[X_center-d1:180, 15:255] != 0)

            # print('ccc3',' 1-9', cancer_num11, cancer_num12, cancer_num13, cancer_num14, cancer_num15,cancer_num16)

            ratio1 = mutation_num2/cancer_num
            ratio2 = mutation_num3 / cancer_num
            ratio3 = mutation_num4 / cancer_num
            ratio4 = mutation_num5 / cancer_num
            # print('cancer number',cancer_num,'\n','mutation ratio2:',ratio1,'\n','mutation ratio3:',ratio2,'\n','mutation ratio4:',ratio3,'\n','mutation ratio5:',ratio4)



            #VAF = [(count_times_element - 1) for count_times_element in range(count_times)] #variant allele frequence


            all_mutation_ID1 = []
            all_mutation_ID2 = []
            all_mutation_ID3 = []
            all_mutation_ID4 = []
            all_mutation_ID5 = []
            all_mutation_ID6 = []
            all_mutation_ID7 = []
            all_mutation_ID8 = []
            all_mutation_ID9 = []
            all_mutation_ID10 = []
            all_mutation_ID11 = []
            all_mutation_ID12 = []
            all_mutation_ID13 = []
            all_mutation_ID14 = []
            all_mutation_ID15 = []
            all_mutation_ID16 = []
            all_mutation_ID17 = []
            all_mutation_ID18 = []
            all_mutation_ID19 = []
            horizon_x11 = []
            horizon_y11 = []
            horizon_x12 = []
            horizon_y12 = []
            horizon_x13 = []
            horizon_y13 = []
            horizon_x14 = []
            horizon_y14 = []
            horizon_x15 = []
            horizon_y15 = []
            horizon_x16 = []
            horizon_y16 = []
            horizon_x17 = []
            horizon_y17 = []
            horizon_x18 = []
            horizon_y18 = []
            horizon_x19 = []
            horizon_y19 = []
            horizon_x1 = []
            horizon_y1 = []
            horizon_x2 = []
            horizon_y2 = []
            horizon_x3 = []
            horizon_y3 = []
            horizon_x4 = []
            horizon_y4 = []
            horizon_x5 = []
            horizon_y5 = []

            vertical_x1 = []
            vertical_y1 = []
            vertical_x2 = []
            vertical_y2 = []
            vertical_x3 = []
            vertical_y3 = []
            vertical_x4 = []
            vertical_y4 = []
            vertical_x5 = []
            vertical_y5 = []



            ### Punch tissue
            cut_size = 0

            X_center = np.random.randint((cut_point1+ cut_size*2**0.5), (cut_point2- cut_size*2**0.5))
            Y_center = np.random.randint((cut_point3+ cut_size*2**0.5), (cut_point4- cut_size*2**0.5))

            cancer_num11 = np.sum(
                cell_matrix[X_center -  cut_size:X_center + 1, Y_center -  cut_size:Y_center + 1] != 0)

            while cancer_num11 != 1:
                X_center = np.random.randint((cut_point1), (cut_point2))
                Y_center = np.random.randint((cut_point3), (cut_point4))
                cancer_num11 = np.sum(
                    cell_matrix[X_center -  cut_size:X_center + 1, Y_center -  cut_size:Y_center + 1] != 0)
            XY_center = (X_center, Y_center)
            XY_range1.append(XY_center)
            for cut in range(0, len(cell_r)):
                x0 = cell_r[cut]
                y0 = cell_c[cut]
                if X_center -  cut_size <= x0 < X_center + 1 and Y_center -  cut_size <= y0 < Y_center + 1:
                    horizon_x11.append(x0)
                    horizon_y11.append(y0)

                mutation_ID = cell_dictionary[x0, y0].mutation  # get all the mutation ID in the dictionary
                # all_mutation_ID += mutation_ID  # Collect every mutation.ID in the all_mutation_ID

            ### 11 slice
            for cut11 in range(0, len(horizon_x11)):
                x11 = horizon_x11[cut11]
                y11 = horizon_y11[cut11]
                mutation_ID11 = cell_dictionary[x11, y11].mutation  # get all the mutation ID in the dictionary

                all_mutation_ID11 += mutation_ID11  # Collect every mutation.ID in the all_mutation_ID1

            result11 = Counter(all_mutation_ID11)  # Count frequency of each mutation ID
            count_mu11 = []  # empty list: record the mutation ID in the VAF
            count_times11 = []  # empty list: record the frequency of mutation ID in the VAF
            # print("n111",  len(all_mutation_ID11))
            ID_key =[]

            for i in result11:

                if result11[i] >= 2*cancer_num11/Filter_frequency:
                    count_mu11.append(i)
                    count_times11.append(result11[i])
                else:
                    continue

            all_mutation_dict1[XY_center] = count_mu11
            # print("nu",len(count_mu11),len(all_mutation_ID11))
            # counter the mutation proberbility in the clip area
            VAF11 = list(map(lambda c: c / (2 * cancer_num11),
                             count_times11))  # Object function change to list, and calculate each element of count_times

            result_VAF11 = Counter(VAF11)  # Count each element in VAF, feedback the numbers

            VAF_mu11 = []  # empty list: record the mutation ID in the VAF
            VAF_times11 = []  # empty list: record the each ID frequency in the VAF
            for i11 in result_VAF11:
                VAF_mu11.append(i11)

            list.sort(VAF_mu11, reverse=1)
            # VAF_mu11 = VAF_mu11[0: (len(VAF_mu11) - 3)]  #remove last 3 point
            # VAF_mu11.remove(0.5)
            # VAF_mu11 = list(filter(lambda x: x >= Filter_frequency, VAF_mu11))
            for j11 in VAF_mu11:
                VAF_times11.append(result_VAF11[j11])

            VAF_cumulative11 = []
            VAF_number11 = list(VAF_times11)
            for f11 in range(1, 1+len(VAF_number11)):
                VAF_cumulative11.append(np.sum(VAF_number11[0:f11]))
            VAF_f11 = list(map(lambda c: 1 / c, VAF_mu11))
            yvals11 = list(map(lambda c: 10 * c-20, VAF_f11))

            all_VAF_f11.append(VAF_f11)
            all_VAF_number11.append(VAF_number11)
            all_VAF_cumulative11.append(VAF_cumulative11)

            "sigle plot for every simulation"
            # plt.figure('Accumulative{}'.format(i_ra), figsize = [5,5],linewidth= 1)
            # trendline(VAF_f11, VAF_cumulative11, VAF_number11, 11, c='b', )
            # plt.scatter(VAF_f11, VAF_cumulative11, s=10)
            # plt.legend(labels='',loc='lower right', shadow=bool, title='Cancer Num= {}'.format(cancer_num11), numpoints=None)
            # plt.plot(VAF_f11, yvals11, 'r', )
            # plt.ylabel("Cumulative Number of Mutation")
            Distance_run1 = ID_run(XY_range1,all_mutation_dict1,cell_dictionary)
            Distance_run2 = Distance_mut_num_run(XY_range1, all_mutation_dict1, cell_dictionary)
        Uni_distance(mutation_type1,XY_range1,all_mutation_dict1,cell_dictionary)
        Uni_distance(mutation_type2, XY_range1, all_mutation_dict1, cell_dictionary)
        Uni_distance(mutation_type3, XY_range1, all_mutation_dict1, cell_dictionary)
        Uni_distance(mutation_type4, XY_range1, all_mutation_dict1, cell_dictionary)
        Uni_distance(mutation_type5, XY_range1, all_mutation_dict1, cell_dictionary)
        Uni_distance(mutation_type6, XY_range1, all_mutation_dict1, cell_dictionary)

        '''add the part of filter algrithm to ignore and analysis the mutation frequency under some value'''
        # for i in result:
        #     count_mu.append(i)
        # for j in count_mu:
        #     count_times.append(result[j])
        # if Heatmap_remove_frequency == True:
        #     remove_mutation = []
        #     for k in range(len(count_times)):
        #         if count_times[k] <= remove_frequency:
        #             remove_mutation.append(count_mu[k])
        #
        #     print('Finis simulation')
        #     for key, cell in list(cell_dictionary.items()):
        #         cell.cp = key
        #         for i in remove_mutation:
        #             if i in cell_dictionary[cell.cp].mutation:
        #                 cell_dictionary[cell.cp].mutation.remove(i)
        #         filter_heatmap[cell.cp] = len(cell_dictionary[cell.cp].mutation)


        a1, b1 = Distance_run1
        a2, b2 = Distance_run2
        Distance_mut_num_boxplot(1, a1, b1)
        Distance_mut_num_boxplot(2,a2, b2)

        # end_time = time.time
        # run_time = end_time - start_time
        #
        # print("time:", run_time)

    '''heatmap segment plot'''
    if Heatmap_whole_plot == True:
        heatmap_aver1 =[]
        heatmap_aver2 = []
        heatmap_aver3 = []
        heatmap_aver4 = []
        heatmap_aver5 = []
        heatmap_aver6 = []

        s1 = []
        s2 = []
        s3 = []
        s4 = []
        s5 = []
        s6 = []
        s7 = []
        s8 = []
        s9 = []
        for i in range(0,heatmap_gener):
            random_point = random.sample(all_cp, 1)

            X_radiation, Y_radiation = random_point[0]

            heatmap_value1=Plot_radiation(1,mutation_type1,10,X_radiation,Y_radiation)
            heatmap_aver1 = np.append(heatmap_aver1, sum(heatmap_value1.tolist(), []), axis=0)

            heatmap_value2=Plot_radiation(2, mutation_type2, 10, X_radiation, Y_radiation)
            heatmap_aver2 = np.append(heatmap_aver2, sum(heatmap_value2.tolist(), []), axis=0)

            heatmap_value3 = Plot_radiation(3, mutation_type3, 10, X_radiation, Y_radiation)
            heatmap_aver3 = np.append(heatmap_aver3, sum(heatmap_value3.tolist(), []), axis=0)

            heatmap_value4 = Plot_radiation(4, mutation_type4, 10, X_radiation, Y_radiation)
            heatmap_aver4 = np.append(heatmap_aver4, sum(heatmap_value4.tolist(), []), axis=0)

            heatmap_value5 = Plot_radiation(5, mutation_type5, 10, X_radiation, Y_radiation)
            heatmap_aver5 = np.append(heatmap_aver5, sum(heatmap_value5.tolist(), []), axis=0)

            heatmap_value6 = Plot_radiation(6, mutation_type6, 10, X_radiation, Y_radiation)
            heatmap_aver6 = np.append(heatmap_aver6, sum(heatmap_value6.tolist(), []), axis=0)

            XY_range1.append((X_radiation,Y_radiation))  #mark the samplling point on cancer tissue

            # heatmap_aver=np.concatenate(heatmap_aver,sum(heatmap_value.tolist(), []),axis=0)

        Map_average(driver_time1[0],mutation_type1,heatmap_aver1)
        Map_average(driver_time2[0],mutation_type2,heatmap_aver2)
        Map_average(driver_time3[0],mutation_type3, heatmap_aver3)
        Map_average(driver_time4[0],mutation_type4, heatmap_aver4)
        Map_average(driver_time5[0],mutation_type5, heatmap_aver5)
        Map_average(driver_time6[0],mutation_type6, heatmap_aver6)
    plt.figure('whole heatmap', figsize=[6, 5])
    plt.title('Pushrate: %s  Cancer Number: {}'.format(cancer_num) % push_rate)
    sns.heatmap(data=cancer_heatmap, yticklabels=False, xticklabels=False, cmap=plt.get_cmap('gist_ncar_r'), )

    plt.figure('Filter heatmap', figsize=[6, 5])
    plt.title('Pushrate: %s  Filter: {}'.format(remove_frequency) % push_rate)
    sns.heatmap(data=filter_heatmap, yticklabels=False, xticklabels=False,cmap= plt.get_cmap('gist_ncar_r'))


    '''fliter hearmap here'''
    # plt.figure('Filter heatmap', figsize=[6, 5])
    # plt.title('Pushrate: %s  Filter Heatmap  Number: {}'.format(cancer_num) % push_rate)
    # sns.heatmap(data=filter_heatmap, yticklabels=False, xticklabels=False, cmap=plt.get_cmap('gist_ncar_r'))


    # whole_heatmap=np.zeros((heatmap_gener,121))
    # for i in range(0,heatmap_gener):
    #     whole_heatmap[i,:] = heatmap_aver[0+121*i:121+121*i]
    # average_heatmap = np.mean(whole_heatmap,axis=0)
    # Average_heatmap = average_heatmap.reshape(11,11)
    # print(average_heatmap)
    #目前只有得出字典的结果，需要把字典的内容平均化


    map = matplotlib.colors.ListedColormap(
        ['#FFFFFF', '#00008B', '#FF0000', '#FFFFFF', '#0000FF', '#6495ED', '#FF7F50', '#ff8d00', '#006400',
         '#8FBC8F', '#9400D3', '#00BFFF', '#FFD700', '#CD5C5C', '#556B2F', '#9932CC', '#8FBC8F', '#2F4F4F',
         '#CD853F', '#FFC0CB', '#4169E1', '#FFFFE0', '#ADD8E6', '#008000', '#9400D3',
         '#9ACD32', '#D2B48C', '#008B8B', '#9400D3', '#00BFFF', '#CD5C5C', ])
    # plt.figure('cancer growth')
    # plt.title('Cancer Number is: {}'.format(cancer_num))
    # plt.imshow(cancer_matrix, cmap=map)
    plt.figure('Cell growth',figsize = [5,5])
    plt.title('Pushrate: %s  Cancer Number: {}'.format(cancer_num) % push_rate)
    plt.imshow(cell_matrix, cmap=map)
    plt.text(10, 15, 'MutationMap', color='black')

    '''sampling point shown in the tissue'''
    # plt.figure('Sampling Cell growth', figsize=[5, 5])
    # plt.title('Pushrate: %s  Cancer Number: {}'.format(cancer_num) % push_rate)
    #
    # """show the every sampling point in the tissue"""
    # for i, j in XY_range1:
    #     plt.text(j, i, 'O', color='red', )
    # for i, j in XY_range2:
    #     plt.text(j, i, '②', color='k')
    # for i, j in XY_range3:
    #     plt.text(j, i, '③', color='g')
    # for i, j in XY_range4:
    #     plt.text(j, i, '④', color='b')
    # for i, j in XY_range5:
    #     plt.text(j, i, '⑤', color='m')
    # for i, j in XY_range6:
    #     plt.text(j, i, '⑥', color='c')
    # plt.imshow(cell_matrix, cmap=map)
    # plt.text(10, 15, 'Sampling size: {}'.format(sampling_size), color='black')

    '''camcer heat map with number of mutation'''
    plt.figure('Cancer heatmap', figsize=[5, 5])
    cancer_heatmap_num = np.sum(cancer_heatmap != 0)
    plt.title('Pushrate: %s  Cancer Number: {}'.format(cancer_heatmap_num) % push_rate)
    plt.text(10, 15, 'HeatMap', color='black', )
    plt.imshow(cancer_heatmap, cmap= plt.get_cmap('hot_r'))
    # mark the heatmap lable in the plot
    for i in range(20,Max_ROW,10):
        value=cancer_heatmap[i,i]
        if value != 0:      #ignore the empty point
            plt.text(i, i, int(value))
    # plt.figure('Average Heatmap', figsize=[6, 6])
    # plt.title('Average Heatmap with {}times samplling'.format(heatmap_gener))
    # plt.imshow(Average_heatmap,cmap= plt.get_cmap('Reds'))
    #
    # for i in range(11):
    #     for j in range(11):
    #         value=Average_heatmap[i,j]
    #         if value != 0:
    #             plt.text(j-0.3, i, value)

    pdf = matplotlib.backends.backend_pdf.PdfPages('filter %s-%s+{}.pdf'.format(cancer_num) % (push_rate,random.randint(1,100)))
    for fig in range(1,11):  ## will open an empty extra figure :(
        pdf.savefig(fig)
    pdf.close()
    if save_mutation == True:
        f = open('max%s+{}.csv'.format(Max_generation) % push_rate, 'a', encoding='utf - 8')
        f.write('{}\n  '.format(cancer_num))
        f.close()
    plt.show()
    plt.ion()




