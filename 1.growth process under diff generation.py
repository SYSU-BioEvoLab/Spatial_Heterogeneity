'''
1. this part shown the tissue distribution of diffrent generation
  [Cancer growth plot]
  [2021/6/14]
this program for tumour tissue growth animation
and plot the figure in the results:
Push rate and Drift dominates early neoplastic dynamics.

  push_rate = 0 Area  500*500      200 generations cancer cell   mutation rate =10
  Plot tumour tissue distributions

 The code using the Object functions and structs to define the ID
 and mutation type of daughter cell, in addition, this part just simulation the situation of cancer growth.
 and show the early mutation distribution in the tissue.
'''

import numpy as np
import math
import random
from scipy import stats
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

Max_generation = 20
# cancer growth space size
Max_ROW = 1200
Max_COL = 1200
push_rate = 1  # the frequency of cancer cell push each others
# growth generation and mark mutation ID in the cancer tissue

### Basic parameter setting
background_gene = 1   # The background gene that cancer originally carried
birth_rate = 1
death_rate = 0
die_divide_times = 500   # The max times of cell divides,  arrived this point the cell will die
Poisson_lambda = 1   # (in generaral means mutation_rate in cancer reseach),the mean value of Poisson distributions

mutation_type1 = 0
mutation_type2 = 3
mutation_type3 = 4
mutation_type4 = 5
mutation_type5 = 6
mutation_type6 = 500

p_color = ['#E2E28A', 'k', '#80E58C', '#4E99D8', '#C1C1C1', '#E080D7','#E28181']  # same color for boxplot and jaccard index


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

mesh_shape = (Max_ROW, Max_COL)     # growth grid
cancer_matrix = np.zeros(mesh_shape)    # create grid space of the cancer growth
cancer_heatmap = np.zeros(mesh_shape)
filter_heatmap = np.zeros(mesh_shape)

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
        if save_data == True:
            f = open('%s+{}.csv'.format(Max_generation) % push_rate, 'a', encoding='utf - 8')
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


'''<<< Jaccard Index >>>'''

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
def Jaccard_func(x, a, b):  # fitting the jaccard curve equation
    return a*(np.power(2,x)**b)

def line_func(x,a,b):
    return a*x+b


def Func_fitting_plot(N,xdata,ydata,fit = False):
    plt.figure('burden_index {}'.format(N), figsize=[5, 5])
    plt.scatter(xdata, ydata, c='k', s=10, alpha=1)
    plt.xlabel('Distance')
    plt.ylabel("Burden gap between point")
    # plt.legend(labels='', loc='upper right', shadow=bool, title='Jaccard Num= {}'.format(len(ydata)), numpoints=None)

    if fit == True:

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
        plt.text(6, max(y_line)-max(y_line)/20, "R^2=%.3f " % Rsqr, color='r')


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
    all_point=(point1,point2,point3,point4,point4)
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
    Func_fitting_plot(20,X_distance,Y_jaccard,fit = True)
    Func_fitting_plot(10, X0, Y0,fit = True)



'''<<< Mutation Burden >>>
calculate the number of mutation in cancer cell,
compare the gap between the diffrent distance'''
def Distance_gap(set_a,set_b):
    D_gap=abs(len(set_a)-len(set_b))
    print('Num:',len(set_a),len(set_b))
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


def Distance_mut_num_run(run_set,all_mutation_dic):
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

def Distance_mut_num_boxplot(cut_dis,X_distance,Y_burden):  #plot th boxplot for diffrent dstance 20 -100
    p1=10 #define the plot position and ±1 bar
    p2=20
    p3=30
    p4=40
    p5=50
    point = [p1,p2,p3,p4,p5]
    point1 = []
    point2 = []
    point3 = []
    point4 = []
    point5 = []
    Y0=[]
    X0=[]
    gap=5
    burden_dict=zip(X_distance, Y_burden)
    for i,j in burden_dict:  #filter burden point in the diffrent distance
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

    labels = [p1, p2, p3,p4, p5,] # for X Axis lable

    point1_mean = np.mean(point1)
    point1_std = np.std(point1,ddof=1)
    fig = plt.figure('Distance_gap_boxplot', figsize=[5, 5])
    ax = fig.add_subplot(111)
    all_point=(point1,point2,point3,point4,point4)
    bplot1 = ax.boxplot(all_point,vert=True,patch_artist = True,  # vertical box alignment
                     labels=labels)
    for box in bplot1['boxes']:
        box.set(color='k',
                linewidth=1)
    for median in bplot1['medians']:
        median.set(color='red',
                   linewidth=1.5)
    plt.xlabel('Distance')
    plt.ylabel("Number variance")
    plt.title('pushrate: %s, filter: {}'.format(remove_frequency) % push_rate)

    # Func_fitting_plot(20,X_distance, Y_burden)
    Func_fitting_plot(20,X_distance,Y_burden,fit = True)
    Func_fitting_plot(10, X0, Y0,fit = True)

def Set_ratio(set_a, set_b):

    driver_number = set_b.count(set_a)

    driver_rate  = driver_number
    return driver_rate
def Driver_mutation(X_center,Y_center):
    mutation_ID = cell_dictionary[X_center,Y_center].mutation
    driver = random.sample(mutation_ID,1)
    return driver

def Inter_set(set_a, set_b):
    intersection_set = list(set(set_a).intersection(set(set_b)))
    return intersection_set



def Radiate_simplling(X_center,Y_center):
    XY_center =(X_center,Y_center)
    all_mutation_ID=[]
    all_mutation_ID11=[]
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

        mutation_ID = cell_dictionary[x0, y0].mutation  # get all the mutation ID in the dictionary
        all_mutation_ID += mutation_ID  # Collect every mutation.ID in the all_mutation_ID
        ### 11 slice
    for cut11 in range(0, len(horizon_x11)):
        x11 = horizon_x11[cut11]
        y11 = horizon_y11[cut11]
        mutation_ID11 = cell_dictionary[x11, y11].mutation  # get all the mutation ID in the dictionary

        all_mutation_ID11 += mutation_ID11  # Collect every mutation.ID in the all_mutation_ID1

    return  all_mutation_ID11

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
        diagnal = binomial(1, 0)
        empty_neighbor_list = [neighbor for neighbor in self.neighbor_pos if neighbor not in agent ]

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
            diagnal_weight = len(diagnal_empty) / (len(diagnal_empty) + 2 * len(square_empty))
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
            poisson1 = Poisson_lambda
            poisson2 = Poisson_lambda
            # print('poisson = ',poisson)
            empty_pos = self.empty_neighbor(agent)      # input empty position
            pushing = push_r()      # input the push rate
            if empty_pos is not None :   # Growth on the empty position
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

                cancer_matrix[self.cp] = agent[self.cp].ip


                cancer_matrix[empty_pos] = daughter_cell.ip

            if empty_pos is None and pushing == 1:

                # Count the number of living cancer around the cell
                new_r, new_c = (0, 1)
                # random choice the directions to pushing
                set_local = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
                new_r, new_c = random_choice(set_local)
                #pushing weight percent (without push_weight in this algrithm)

                cp_r, cp_c = self.cp
                newcell_pos = (cp_r + new_r, cp_c + new_c)
                newcell2_pos = (cp_r + new_r + new_r, cp_c + new_c + new_c)
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
                all_ip.append(agent[self.cp].ip)



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




            '''the cancer cell death process'''
            # the cell death algorithm (spontaneous death) and (divide death)
            self.die_divide_times -= 1   # The life span of a cell divided is reduced by one
            spontaneous_death = death_r()
            # (add divide death code: 'or self.die_divide_times <= 0')
            if spontaneous_death == 1:
                cancer_matrix[self.cp] = 0
                agent[self.cp].mutation = [-1]
                all_mutation_del.append(self.cp)


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

    for rep in range(cell_replication):   # replicate cell in the generation
        cell_list = list(cell_dictionary.items())
        for ID in range(len(cell_list)):
            tissue_dictionary.update({ID: list(cell_dictionary.items())[ID]})
        grow_list = list(mutations_dictionary.keys())
        # get the total cell in list

        # print(grow_list)

        shuffle(grow_list)
        for key in  grow_list:  # Iterate through cell_dictionary to find the center point
            cp = mutations_dictionary[key]
            cell = cell_dictionary[cp]
            cell.cp = cp
            c_r, c_c = cell.cp
            cell_dictionary[cell.cp].mutation = cell.mutation   # Assign the mutation ID to each center point
            cancer_matrix[cell.cp] = cell.ip  # Assign the ID to each center point

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
        cancer_num = np.sum(cell_matrix != 0)
        map = matplotlib.colors.ListedColormap(
            ['#FFFFFF', '#00008B', '#FF0000', '#FFFFFF', '#0000FF', '#6495ED', '#FF7F50', '#ff8d00', '#006400',
             '#8FBC8F', '#9400D3', '#00BFFF', '#FFD700', '#CD5C5C', '#556B2F', '#9932CC', '#8FBC8F', '#2F4F4F',
             '#CD853F', '#FFC0CB', '#4169E1', '#FFFFE0', '#ADD8E6', '#008000', '#9400D3',
             '#9ACD32', '#D2B48C', '#008B8B', '#9400D3', '#00BFFF', '#CD5C5C', ])
        plt.figure('Cell growth %s'% rep, figsize=[5, 5])
        plt.title('Pushrate: %s  Cancer Number: {}'.format(cancer_num) % push_rate)
        plt.imshow(cell_matrix, cmap= map)
        plt.text(10, 15, 'Cell Generation:{}'.format(rep+1), color='black')

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
    plt.text(10, 15, 'MutationMap',color='black')

    pdf = matplotlib.backends.backend_pdf.PdfPages('Growthplot %s-%s+{}.pdf'.format(cancer_num) % (push_rate,random.randint(1,100)))
    for fig in range(1,Max_generation+2):  ## will open an empty extra figure :(
        pdf.savefig(fig)
    pdf.close()
    plt.show()
    plt.ion()




