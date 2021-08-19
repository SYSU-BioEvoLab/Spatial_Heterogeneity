'''
[Cancer growth model]
  [2021/7/28]
    This part show  the Jaccard_index and accumulative
    meantime cut the different area, from 100 - 3600
    analysed the VAF and fitting the accumulative point.
    therefore, plot the Jaccard index point in the whole figure.

'''

import numpy as np
import math
import random
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors   # define the colour by myself
import matplotlib.backends.backend_pdf
import pandas as pd    # Applied to cell pushing
from numba import jit
import time     # calculate the spend time of code
from collections import Counter     # count the frequency of mutation{{}
from itertools import combinations  # appy jaccard index set combian each other
start_time = time.time()
### Main parameter setting
Max_generation = 13
push_rate = 1
Max_ROW = 400
Max_COL = 400
sample_number = 50
Filter_frequency = 50 #filter the allele frequency less than this value.


### Basic parameter setting
background_gene = 1   # The background gene that cancer originally carried
  # the frequency of cancer cell push each others
birth_rate = 1  #
death_rate = 0
# birth_rate = int(input('Birth rate ='))
# death_rate = int(input('death rate ='))  # The death probability of cell growth process
die_divide_times = 500   # The max times of cell divides,  arrived this point the cell will die
Poisson_lambda = 10   # (in generaral means mutation_rate in cancer reseach),the mean value of Poisson distributions


mutation_type2 = 0
mutation_type3 = 10
mutation_type4 = 30
mutation_type5 = 50



### Show the fellowing mutation ID in the cencer tissue
global all_mutation
global list_newcell
global all_ip
global list_mutation
global less_list
list_newcell =[]
list_newcell2 =[]

global jacard_dict
jacard_dict = {}

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
pcolor = ['#6495ED', '#FF7F50', '#6e6ee5', '#5bd85b',
         '#E56ADC', '#9400D3', '#3FC5E8', '#FFD700', '#CD5C5C', '#556B2F', '#9932CC', '#8FBC8F', '#2F4F4F',
         '#CD853F', '#FFC0CB', '#469E1', '#FFFFE0', '#ADD8E6', '#008000', '#9400D3',
         '#9ACD32', '#D2B48C', '#008B8B', '#9400D3', '#00BFFF', '#CD5C5C',]
# mutation_type5 = 55
# mutation_type6 = 0

mesh_shape = (Max_ROW, Max_COL)     # growth grid
cancer_matrix = np.zeros(mesh_shape)    # create grid space of the cancer growth
f = open('0.1+{}.csv'.format(Max_generation), 'a', encoding='utf - 8')
f.write('Slope  ')
f.write('R^2  ')
f.write('Ks\n')
f.close()

def cancer_mesh(row,col):    # define cancer growth mesh size
    all_mesh = [(r,c)for r in np.arange(0,row) for c in np.arange(0, col)]
    central_pos = {cp:neighbor_mesh(cp,row,col) for cp in all_mesh}  # define the center position
    return central_pos
@jit
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
def fit_func(x,a,b):
    return a*(x-1/0.495)

def trendline(xd, yd,zd, term, order=1, c='r', alpha=0.2, Rval=False):
    """Make a line of best fit"""

    popt, pcov = curve_fit(fit_func, xd, yd)
    slope = popt[0]

    # yvals = fit_func(xd, a)  # 拟合y值


    #Calculate trendline
    coeffs = np.polyfit(xd, yd, order)

    intercept = coeffs[-1]
    slope = coeffs[-2]
    power = coeffs[0] if order == 2 else 0
    # print('slope',slope)
    minxd = np.min(xd)
    maxxd = np.max(xd)

    xl = np.array([minxd, maxxd])
    yl =  slope * xl
    ya = slope * (xl-1/0.495)
    y1 = stats.ks_2samp(zd, yl)

    #Plot trendline
    # plt.plot(xl, yl,'orange', alpha=1)
    plt.plot(xl, ya, 'b',alpha=0.1)

    #Calculate R Squared
    parameter = np.poly1d(coeffs)

    ybar = np.sum(yd) / len(yd)
    ssreg = np.sum((parameter(xd) - ybar) ** 2)
    sstot = np.sum((yd - ybar) ** 2)
    Rsqr = ssreg / sstot

    # Kolmogorov-Smirnov test
    Data_dic = {}
    K_distance = []
    KS_dict ={}
    for i in range(len(xd)):
        Data_dic[xd[i]] = yd[i]
    for i in xd:
        K_gap = abs(Data_dic[i] - (slope * (i - 1/0.495)))
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
        print('slope{}'.format(term),slope, '$R^2 = %0.3f' % Rsqr,'K-S',KS)
        # print('slope{}'.format(term), slope, 'K-S',KS)
        f = open('ks+{}.csv'.format(term), 'a', encoding='utf - 8')


        f.write('{},'.format(round(slope, 2)))
        f.write('{},'.format(round(Rsqr, 3)))

        f.write('{}\n'.format(round(KS, 3)))
        f.close()
def Distance_KS(xd, yd):
    popt, pcov = curve_fit(fit_func, xd, yd)
    a = 10
    Data_dic = {}
    K_distance = []
    KS_dict ={}
    for i in range(len(xd)):
        Data_dic[xd[i]] = yd[i]
    for i in xd:
        K_gap = abs(Data_dic[i] - (a * (i - 1/0.495)))
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
def len_intersection(set_a, set_b):
    intersection_set = list(set(set_a).intersection(set(set_b)))
    len_intersection = len(intersection_set)
    return len_intersection
def len_union(set_a, set_b):
    intersection_set = list(set(set_a).union(set(set_b)))
    len_intersection = len(intersection_set)
    return len_intersection
def combine(temp_list, n):
    '''根据n获得列表中的所有可能组合（n个元素为一组）'''
    temp_list2 = []
    for c in combinations(temp_list, n):
        temp_list2.append(c)
    return temp_list2
def jacard_run(i_cut,run_set,all_mutation_dic):
    combine_set = combine(run_set,2)

    X_distance = []
    Y_jaccard = []


    for i, j in combine_set:
        i_x,i_y = i
        j_x, j_y = j
        D_i_j = math.sqrt((j_x-i_x)**2+(j_y-i_y)**2)   # calculate the distance of two combination points

        X_distance.append(D_i_j)
        jacard_dict[D_i_j]= jacard_index(all_mutation_dic[i],all_mutation_dic[j])
        Y_jaccard.append(jacard_index(all_mutation_dic[i],all_mutation_dic[j]))

    return X_distance,Y_jaccard
    #     print('assssss',i, j,D_i_j,jacard_index(all_mutation_dict1[i],all_mutation_dict3[j]), )
    # print('zzz',math.factorial(10),len(X_distance),len(Y_jaccard))
def Jacard_boxplot( n,Y_jaccard,X_distance,):
    p1=20
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
    p_color = ['#66C2A5',  '#FC8C62', '#8DA0CB', '#E78AC3', '#A6D854','#FFD92F', '#7ECECA', '#CD5C5C', '#556B2F', '#9932CC', '#8FBC8F', '#2F4F4F',
         '#CD853F', '#FFC0CB', '#469E1', '#FFFFE0', '#ADD8E6', '#008000', '#9400D3',
         '#9ACD32', '#D2B48C', '#008B8B']


    for i in X_distance:
        print(i)
        if p1-1 <= i <= p1+1:
            point1.append(jacard_dict[i])
        elif p2-1 <= i <= p2+1:
            point2.append(jacard_dict[i])
        elif p3 - 1 <= i <= p3 + 1:
            point3.append(jacard_dict[i])
        elif p4 - 1 <= i <= p4 + 1:
            point4.append(jacard_dict[i])
        elif p5 - 1 <= i <= p5 + 1:
            point5.append(jacard_dict[i])

    labels = ['20', '40', '60','80', '100',]
    point1_mean = np.mean(point1)
    point1_std = np.std(point1,ddof=1)
    all_point = (point1, point2, point3, point4, point5)
    plt.figure('jacard_SNS_all', figsize=[5, 5])

    sns.boxplot(data=all_point,color=matplotlib.cm.Set2(n),linewidth=0.5)
    plt.xlabel('Distance')
    plt.xticks(range(0,5),labels)
    plt.ylabel("Jaccard index")

    plt.figure('jacard_poin all' , figsize=[5, 5])
    sns.swarmplot(data=all_point, color=matplotlib.cm.Set2(n))
    plt.xlabel('Distance')
    plt.ylabel("Jaccard index")
    plt.xticks(range(0,5),labels)

    plt.figure('jacard_Violin_all', figsize=[5, 5])
    sns.violinplot(data=all_point, notch=True, vert=True, patch_artist=True, alpha=0.5, # vertical box alignment
                   xlabels=labels, palette="Set2", linewidth=0.2, inner=None)
    sns.swarmplot(data=all_point, color=".1",dodge=False,size= 3,alpha=0.5)
    plt.xlabel('Distance')
    plt.ylabel("Jaccard index")
    plt.xticks(range(0, 5), labels)

    plt.figure('jacard_Violin%s' % n, figsize=[5, 5])
    sns.violinplot(data=all_point,notch=True,vert=True,patch_artist = True,  # vertical box alignment
                     xlabels=labels,palette="Set2",linewidth=0.5, inner=None)
    sns.swarmplot(data=all_point, color=".1",alpha = 0.5,dodge=True,size= 3)
    plt.xlabel('Distance')
    plt.ylabel("Jaccard index")
    plt.xticks(range(0,5),labels)



    plt.figure('jacard_SNS%s' % n, figsize=[5, 5])
    sns.boxplot( data=all_point,palette="Set2",linewidth=0.5,width=0.75)
    sns.swarmplot(data=all_point, color=".1",size= 3,alpha = 0.5)
    plt.xlabel('Distance')
    plt.ylabel("Jaccard index")
    plt.xticks(range(0,5),labels)
    fig = plt.figure('jacard_boxplot_all', figsize=[5, 5])

    ax = fig.add_subplot(111)

    bplot1 = ax.boxplot(all_point, vert=True,patch_artist=True, # vertical box alignment
                        labels=labels)
    for box in bplot1['boxes']:
        box.set(color=matplotlib.cm.Set2(n),
                linewidth=1,alpha = 0.5)
    for median in bplot1['medians']:
        median.set(color='red',
                   linewidth=1.5)
    plt.legend(loc='upper right', shadow=bool, title='Jaccard Num= {}'.format(len(Y_jaccard)),
               numpoints=None)
    plt.xlabel('Distance')
    plt.ylabel("Jaccard index")


    fig = plt.figure('jacard_boxplot%s'%n, figsize=[5, 5])

    ax = fig.add_subplot(111)


    bplot1 = ax.boxplot(all_point,vert=True,patch_artist = True,# vertical box alignment
                     labels=labels)
    for box in bplot1['boxes']:
        box.set(color=matplotlib.cm.Set2(n),
                linewidth=1,alpha=0.5)
    for median in bplot1['medians']:
        median.set(color='red',
                   linewidth=1.5)

    plt.figure('jacard_index_cut3', figsize=[5, 5])
    plt.scatter(X_distance, Y_jaccard, c=pcolor[n], s=10, alpha=0.8)
    plt.xlabel('Distance')
    plt.ylabel("Jaccard index")
    plt.legend(labels='', loc='upper right', shadow=bool, title='Jaccard Num= {}'.format(len(Y_jaccard)),
               numpoints=None)

    plt.figure('jacard_index_cut2', figsize=[5, 5])
    plt.scatter(X_distance, Y_jaccard, c=p_color[n], s=10, alpha=1)
    plt.xlabel('Distance')
    plt.ylabel("Jaccard index")
    plt.legend(labels='', loc='upper right', shadow=bool, title='Jaccard Num= {}'.format(len(Y_jaccard)),
               numpoints=None)

    plt.figure('jacard_index_cut', figsize=[5, 5])
    plt.scatter(X_distance, Y_jaccard, c=pcolor[n],s=10, alpha=0.5)
    plt.xlabel('Distance')
    plt.ylabel("Jaccard index")
    plt.legend(labels = '',loc='upper right', shadow=bool, title='Jaccard Num= {}'.format(len(Y_jaccard)),
               numpoints=None)
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
            poisson1 = Poisson()
            poisson2 = Poisson()
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
                    agent[self.cp].ID =2
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
                all_ip.append(agent[self.cp].ip)

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
        cell.cp = key
        c_r, c_c = cell.cp
        # print('cp', cp, cell.mutation)

        cell_dictionary[cell.cp].mutation = cell.mutation
        list_mutation.append(cell.cp)

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

        gap = round((cut_point2 - cut_point1) / 5)
        gap2 = round((cut_point4 - cut_point3) / 5)
    # print('ccc', gap, cut_point, cut_point1, cut_point2)

    XY_range1 =[]
    XY_range2 = []
    XY_range3 = []
    XY_range4 = []
    XY_range5 = []
    XY_range6 = []

    all_VAF_f1 =[]
    all_VAF_f2 = []
    all_VAF_f3 = []
    all_VAF_f4 = []
    all_VAF_f5 = []
    all_VAF_f6 = []

    all_VAF_mu1 =[]
    all_VAF_mu2 = []
    all_VAF_mu3 = []
    all_VAF_mu4 = []
    all_VAF_mu5 = []
    all_VAF_mu6 = []

    all_mutation_ID = []
    all_VAF_number1 = []
    all_VAF_number2 = []
    all_VAF_number3 = []
    all_VAF_number4 = []
    all_VAF_number5 = []
    all_VAF_number6 = []
    all_VAF_cumulative1 = []
    all_VAF_cumulative2 = []
    all_VAF_cumulative3 = []
    all_VAF_cumulative4 = []
    all_VAF_cumulative5 = []
    all_VAF_cumulative6 = []
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
    # counter the mutation in the clip area
    result = Counter(all_mutation_ID)  # Count frequency of each mutation ID
    count_mu = []  # empty list: record the mutation ID in the VAF
    count_times = []  # empty list: record the frequency of mutation ID in the VAF

    for i in result:
        count_mu.append(i)
    for j in count_mu:
        count_times.append(result[j])
    # counter the mutation proberbility in the clip area
    VAF = list(map(lambda n: n / (2 * cancer_num),
                   count_times))  # Object function change to list, and calculate each element of count_times
    result_VAF = (Counter(VAF))  # Count each element in VAF, feedback the numbers
    VAF_mu = []  # empty list: record the mutation ID in the VAF
    VAF_times = []  # empty list: record the each ID frequency in the VAF
    for i1 in result_VAF:
        VAF_mu.append(i1)
    list.sort(VAF_mu, reverse=1)
    for j1 in VAF_mu:
        VAF_times.append(result_VAF[j1])
    VAF_cumulative = []

    VAF_number = list(VAF_times)
    for f in range(1, len(VAF_number) + 1):
        VAF_cumulative.append(np.sum(VAF_number[0:f]))
    VAF_f = list(map(lambda c: 1 / c, VAF_mu))

    yvals = list(map(lambda c: 10 * c-20, VAF_f))


    for i_cut in range(1,7):
        for i_ra in range(1,sample_number +1 ):


            distance = 5
            d1 = 5
            d2 = d1+5
            d3 = d1+10
            d4 = d1+15
            d5 = d1+20
            d6 = d1+25

            # cut the tissue into 5 piece



            # cancer_num17 = np.sum(cell_matrix[X_center-d1:X_center+d1, 5:255] != 0)
            # cancer_num18 = np.sum(cell_matrix[X_center-d1:175, 5:255] != 0)
            # cancer_num19 = np.sum(cell_matrix[X_center-d1:180, 5:255] != 0)

            # print('ccc3',' 1-9', cancer_num1, cancer_num2, cancer_num3, cancer_num4, cancer_num5,cancer_num6)

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
            all_mutation_ID1 = []
            all_mutation_ID2 = []
            all_mutation_ID3 = []
            all_mutation_ID4 = []
            all_mutation_ID5 = []
            all_mutation_ID6 = []
            all_mutation_ID17 = []
            all_mutation_ID18 = []
            all_mutation_ID19 = []
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
            horizon_x6 = []
            horizon_y6 = []
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
            if i_cut == 1:

                X_center = np.random.randint((cut_point1+d1*2**0.5), (cut_point2-d1*2**0.5))
                Y_center = np.random.randint((cut_point3+d1*2**0.5), (cut_point4-d1*2**0.5))

                cancer_num1 = np.sum(
                    cell_matrix[X_center - d1:X_center + d1, Y_center - d1:Y_center + d1] != 0)

                while cancer_num1 != 100:
                    X_center = np.random.randint((cut_point1), (cut_point2))
                    Y_center = np.random.randint((cut_point3), (cut_point4))
                    cancer_num1 = np.sum(
                        cell_matrix[X_center - d1:X_center + d1, Y_center - d1:Y_center + d1] != 0)
                XY_center = (X_center, Y_center)
                XY_range1.append(XY_center)
                for cut in range(0, len(cell_r)):
                    x0 = cell_r[cut]
                    y0 = cell_c[cut]
                    if X_center - d1 <= x0 < X_center + d1 and Y_center - d1 <= y0 < Y_center + d1:
                        horizon_x1.append(x0)
                        horizon_y1.append(y0)

                    mutation_ID = cell_dictionary[x0, y0].mutation  # get all the mutation ID in the dictionary
                    all_mutation_ID += mutation_ID  # Collect every mutation.ID in the all_mutation_ID

                ### 1 slice
                for cut1 in range(0, len(horizon_x1)):
                    x1 = horizon_x1[cut1]
                    y1 = horizon_y1[cut1]
                    mutation_ID1 = cell_dictionary[x1, y1].mutation  # get all the mutation ID in the dictionary

                    all_mutation_ID1 += mutation_ID1  # Collect every mutation.ID in the all_mutation_ID1

                result1 = Counter(all_mutation_ID1)  # Count frequency of each mutation ID
                count_mu1 = []  # empty list: record the mutation ID in the VAF
                count_times1 = []  # empty list: record the frequency of mutation ID in the VAF
                # print("n1",  len(all_mutation_ID1))
                ID_key =[]

                for i in result1:

                    if result1[i] >= cancer_num1/Filter_frequency:
                        count_mu1.append(i)
                        count_times1.append(result1[i])
                    else:
                        continue

                all_mutation_dict1[XY_center] = count_mu1
                # print("nu",len(count_mu1),len(all_mutation_ID1))
                # counter the mutation proberbility in the clip area
                VAF1 = list(map(lambda c: c / (cancer_num1),
                                 count_times1))  # Object function change to list, and calculate each element of count_times

                result_VAF1 = Counter(VAF1)  # Count each element in VAF, feedback the numbers

                VAF_mu1 = []  # empty list: record the mutation ID in the VAF
                VAF_times1 = []  # empty list: record the each ID frequency in the VAF
                for i1 in result_VAF1:
                    VAF_mu1.append(i1)

                list.sort(VAF_mu1, reverse=1)
                # VAF_mu1 = VAF_mu1[0: (len(VAF_mu1) - 3)]  #remove last 3 point
                # VAF_mu1.remove(0.5)
                # VAF_mu1 = list(filter(lambda x: x >= Filter_frequency, VAF_mu1))
                for j1 in VAF_mu1:
                    VAF_times1.append(result_VAF1[j1])

                VAF_cumulative1 = []
                VAF_number1 = list(VAF_times1)
                for f1 in range(1, 1+len(VAF_number1)):
                    VAF_cumulative1.append(np.sum(VAF_number1[0:f1]))
                VAF_f1 = list(map(lambda c: 1 / c, VAF_mu1))
                yvals1 = list(map(lambda c: 10 * c-20, VAF_f1))
                all_VAF_mu1.append(VAF_mu1)
                all_VAF_f1.append(VAF_f1)
                all_VAF_number1.append(VAF_number1)
                all_VAF_cumulative1.append(VAF_cumulative1)

                "sigle plot for every simulation"
                # plt.figure('Accumulative{}'.format(i_ra), figsize = [5,5],linewidth= 1)
                # trendline(VAF_f1, VAF_cumulative1, VAF_number1, 1, c='b', )
                # plt.scatter(VAF_f1, VAF_cumulative1, s=10)
                # plt.legend(labels='',loc='lower right', shadow=bool, title='Cancer Num= {}'.format(cancer_num1), numpoints=None)
                # plt.plot(VAF_f1, yvals1, 'r', )
                # plt.ylabel("Cumulative Number of Mutation")
                jacard_run1=jacard_run(i_cut,XY_range1,all_mutation_dict1)

            ### 2 slice
            if i_cut == 2:
                X_center = np.random.randint((cut_point1 + d2*2**0.5), (cut_point2 - d2*2**0.5))
                Y_center = np.random.randint((cut_point3 + d2*2**0.5), (cut_point4 - d2*2**0.5))

                cancer_num2 = np.sum(cell_matrix[X_center - d2:X_center + d2, Y_center - d2:Y_center + d2] != 0)
                while cancer_num2 != 400:
                    X_center = np.random.randint((cut_point1 + d2*2**0.5), (cut_point2 - d2*2**0.5))
                    Y_center = np.random.randint((cut_point3 + d2*2**0.5), (cut_point4 - d2*2**0.5))
                    cancer_num2 = np.sum(
                        cell_matrix[X_center - d2:X_center + d2, Y_center - d2:Y_center + d2] != 0)
                XY_center = (X_center, Y_center)
                XY_range2.append(XY_center)

                for cut in range(0, len(cell_r)):
                    x0 = cell_r[cut]
                    y0 = cell_c[cut]

                    if X_center - d2 <= x0 < X_center + d2 and Y_center - d2 <= y0 < Y_center + d2:
                        horizon_x2.append(x0)
                        horizon_y2.append(y0)


                    mutation_ID = cell_dictionary[x0, y0].mutation  # get all the mutation ID in the dictionary
                    all_mutation_ID += mutation_ID  # Collect every mutation.ID in the all_mutation_ID
                for cut2 in range(0, len(horizon_x2)):
                    x2 = horizon_x2[cut2]
                    y2 = horizon_y2[cut2]
                    mutation_ID2 = cell_dictionary[x2, y2].mutation  # get all the mutation ID in the dictionary

                    all_mutation_ID2 += mutation_ID2  # Collect every mutation.ID in the all_mutation_ID1



                result2 = Counter(all_mutation_ID2)  # Count frequency of each mutation ID
                count_mu2 = []  # empty list: record the mutation ID in the VAF
                count_times2 = []  # empty list: record the frequency of mutation ID in the VAF
                show_ID2={}
                for i in result2:
                    if result2[i] >= cancer_num2/Filter_frequency:
                        count_mu2.append(i)
                        count_times2.append(result2[i])
                    else:
                        continue

                all_mutation_dict2[XY_center] = count_mu2
                # counter the mutation proberbility in the clip area
                VAF2 = list(map(lambda c: c / (cancer_num2),
                                 count_times2))  # Object function change to list, and calculate each element of count_times

                result_VAF2 = Counter(VAF2)  # Count each element in VAF, feedback the numbers

                VAF_mu2 = []  # empty list: record the mutation ID in the VAF
                VAF_times2 = []  # empty list: record the each ID frequency in the VAF
                for i2 in result_VAF2:
                    VAF_mu2.append(i2)
                list.sort(VAF_mu2, reverse=1)
                # VAF_mu2 = VAF_mu2[0: (len(VAF_mu2) - 3)]
                # ## 2020.7.7 filter the result less than 50
                # VAF_mu2 = list(filter(lambda x: x >= Filter_frequency, VAF_mu2))
                # VAF_mu2.remove(0.5)

                for j2 in VAF_mu2:
                    VAF_times2.append(result_VAF2[j2])

                VAF_cumulative2 = []
                VAF_number2 = list(VAF_times2)
                for f2 in range(1,1+len(VAF_number2)):
                    VAF_cumulative2.append(np.sum(VAF_number2[0:f2]))
                VAF_f2 = list(map(lambda c: 1 / c, VAF_mu2))
                yvals2 = list(map(lambda c: 10 * c-20, VAF_f2))

                all_VAF_mu2.append(VAF_mu2)
                all_VAF_f2.append(VAF_f2)
                all_VAF_number2.append(VAF_number2)
                all_VAF_cumulative2.append(VAF_cumulative2)
                "sigle plot for every simulation"
                # plt.figure('2Accumulative{}'.format(i_ra), figsize=[5, 5], linewidth=1)
                # trendline(VAF_f2, VAF_cumulative2, VAF_number2, 2, c='b')
                # plt.scatter(VAF_f2, VAF_cumulative2, s=10)
                # plt.legend(labels='',loc='lower right', shadow=bool, title='Cancer Num= {}'.format(cancer_num2), numpoints=None)
                # plt.plot(VAF_f2, yvals2, 'r', )
                # plt.ylabel("Cumulative Number of Mutation")
                jacard_run2 = jacard_run(i_cut,XY_range2, all_mutation_dict2)

            ## 3 slice
            if i_cut == 3:
                X_center = np.random.randint((cut_point1 + d3*2**0.5), (cut_point2 - d3*2**0.5))
                Y_center = np.random.randint((cut_point3 + d3*2**0.5), (cut_point4 - d3*2**0.5))
                cancer_num3 = np.sum(cell_matrix[X_center - d3:X_center + d3, Y_center - d3:Y_center + d3] != 0)

                while cancer_num3 != 900:
                    X_center = np.random.randint((cut_point1 + d3*2**0.5), (cut_point2 - d3*2**0.5))
                    Y_center = np.random.randint((cut_point3 + d3*2**0.5), (cut_point4 - d3*2**0.5))
                    cancer_num3 = np.sum(
                        cell_matrix[X_center - d3:X_center + d3, Y_center - d3:Y_center + d3] != 0)
                XY_center = (X_center, Y_center)
                XY_range3.append(XY_center)
                cancer_num3 = np.sum(cell_matrix[X_center - d3:X_center + d3, Y_center - d3:Y_center + d3] != 0)

                for cut in range(0, len(cell_r)):
                    x0 = cell_r[cut]
                    y0 = cell_c[cut]

                    if X_center - d3 <= x0 < X_center + d3 and Y_center - d3 <= y0 < Y_center + d3:
                        horizon_x3.append(x0)
                        horizon_y3.append(y0)


                    mutation_ID = cell_dictionary[x0, y0].mutation  # get all the mutation ID in the dictionary
                    all_mutation_ID += mutation_ID  # Collect every mutation.ID in the all_mutation_ID

                for cut3 in range(0, len(horizon_x3)):
                    x3 = horizon_x3[cut3]
                    y3 = horizon_y3[cut3]
                    mutation_ID3 = cell_dictionary[x3, y3].mutation  # get all the mutation ID in the dictionary

                    all_mutation_ID3 += mutation_ID3  # Collect every mutation.ID in the all_mutation_ID1


                result3 = Counter(all_mutation_ID3)  # Count frequency of each mutation ID
                count_mu3 = []  # empty list: record the mutation ID in the VAF
                count_times3 = []  # empty list: record the frequency of mutation ID in the VAF

                for i in result3:
                    if result3[i] >= cancer_num3 / Filter_frequency:
                        count_mu3.append(i)
                        count_times3.append(result3[i])
                    else:
                        continue

                all_mutation_dict3[XY_center] = count_mu3
                # counter the mutation proberbility in the clip area
                VAF3 = list(map(lambda c: c / ( cancer_num3),
                                 count_times3))  # Object function change to list, and calculate each element of count_times

                result_VAF3 = Counter(VAF3)  # Count each element in VAF, feedback the numbers

                VAF_mu3 = []  # empty list: record the mutation ID in the VAF
                VAF_times3 = []  # empty list: record the each ID frequency in the VAF
                for i3 in result_VAF3:
                    VAF_mu3.append(i3)
                list.sort(VAF_mu3, reverse=1)
                # VAF_mu3 = VAF_mu3[0: (len(VAF_mu3) - 3)]
                # ## 2020.7.7 filter the result less than 50
                # VAF_mu3 = list(filter(lambda x: x >= Filter_frequency, VAF_mu3))
                # VAF_mu3.remove(0.5)
                for j3 in VAF_mu3:
                    VAF_times3.append(result_VAF3[j3])

                VAF_cumulative3 = []
                VAF_number3 = list(VAF_times3)
                for f3 in range(1,1+len(VAF_number3)):
                    VAF_cumulative3.append(np.sum(VAF_number3[0:f3]))
                VAF_f3 = list(map(lambda c: 1 / c, VAF_mu3))
                yvals3 = list(map(lambda c: 10 * c-20, VAF_f3))
                all_VAF_cumulative3.append(VAF_cumulative3)
                all_VAF_number3.append(VAF_number3)
                all_VAF_f3.append(VAF_f3)
                all_VAF_mu3.append(VAF_mu3)
                "sigle plot for every simulation"
                # plt.figure('3Accumulative{}'.format(i_ra), figsize=[5, 5], linewidth=1)
                # trendline(VAF_f3, VAF_cumulative3, VAF_number3, 3, c='b',)
                # plt.scatter(VAF_f3, VAF_cumulative3, s=10)
                # plt.legend(labels='',loc='lower right', shadow=bool, title='Cancer Num= {}'.format(cancer_num3), numpoints=None)
                # plt.plot(VAF_f3, yvals3, 'r', )
                # plt.ylabel("Cumulative Number of Mutation")
                jacard_run3=jacard_run(i_cut,XY_range3, all_mutation_dict3)
            # 4 slice
            if i_cut == 4:
                X_center = np.random.randint((cut_point1 + d4*2**0.5), (cut_point2 - d4*2**0.5))
                Y_center = np.random.randint((cut_point3 + d4*2**0.5), (cut_point4 - d4*2**0.5))
                cancer_num4 = np.sum(cell_matrix[X_center - d4:X_center + d4, Y_center - d4:Y_center + d4] != 0)
                while cancer_num4 != 1600:
                    X_center = np.random.randint((cut_point1 + d4 * 2 ** 0.5), (cut_point2 - d4 * 2 ** 0.5))
                    Y_center = np.random.randint((cut_point3 + d4 * 2 ** 0.5), (cut_point4 - d4 * 2 ** 0.5))
                    cancer_num4 = np.sum(
                        cell_matrix[X_center - d4:X_center + d4, Y_center - d4:Y_center + d4] != 0)
                XY_center = (X_center, Y_center)
                XY_range4.append(XY_center)

                cancer_num4 = np.sum( cell_matrix[X_center - d4:X_center + d4, Y_center - d4:Y_center + d4] != 0)

                for cut in range(0, len(cell_r)):
                    x0 = cell_r[cut]
                    y0 = cell_c[cut]

                    if X_center - d4 <= x0 < X_center + d4 and Y_center - d4 <= y0 < Y_center + d4:
                        horizon_x4.append(x0)
                        horizon_y4.append(y0)


                    mutation_ID = cell_dictionary[x0, y0].mutation  # get all the mutation ID in the dictionary
                    all_mutation_ID += mutation_ID  # Collect every mutation.ID in the all_mutation_ID
                for cut4 in range(0, len(horizon_x4)):
                    x4 = horizon_x4[cut4]
                    y4 = horizon_y4[cut4]
                    mutation_ID4 = cell_dictionary[x4, y4].mutation  # get all the mutation ID in the dictionary

                    all_mutation_ID4 += mutation_ID4  # Collect every mutation.ID in the all_mutation_ID1

                result4 = Counter(all_mutation_ID4)  # Count frequency of each mutation ID
                count_mu4 = []  # empty list: record the mutation ID in the VAF
                count_times4 = []  # empty list: record the frequency of mutation ID in the VAF
                for i in result4:
                    if result4[i] >= cancer_num4 / Filter_frequency:
                        count_mu4.append(i)
                        count_times4.append(result4[i])
                    else:
                        continue
                all_mutation_dict4[XY_center] = count_mu4
                # counter the mutation proberbility in the clip area
                VAF4 = list(map(lambda c: c / ( cancer_num4),
                                 count_times4))  # Object function change to list, and calculate each element of count_times

                result_VAF4 = Counter(VAF4)  # Count each element in VAF, feedback the numbers

                VAF_mu4 = []  # empty list: record the mutation ID in the VAF
                VAF_times4 = []  # empty list: record the each ID frequency in the VAF
                for i4 in result_VAF4:
                    VAF_mu4.append(i4)
                list.sort(VAF_mu4, reverse=1)
                # VAF_mu4 = VAF_mu4[0: (len(VAF_mu4) - 3)]
                # ## 2020.7.7 filter the result less than 50
                # VAF_mu4 = list(filter(lambda x: x >= Filter_frequency, VAF_mu4))
                # VAF_mu4.remove(0.5)
                for j4 in VAF_mu4:
                    VAF_times4.append(result_VAF4[j4])

                VAF_cumulative4 = []
                VAF_number4 = list(VAF_times4)
                for f4 in range(1,1+len(VAF_number4)):
                    VAF_cumulative4.append(np.sum(VAF_number4[0:f4]))
                VAF_f4 = list(map(lambda c: 1 / c, VAF_mu4))
                yvals4 = list(map(lambda c: 10 * c-20, VAF_f4))
                all_VAF_cumulative4.append(VAF_cumulative4)
                all_VAF_number4.append(VAF_number4)
                all_VAF_f4.append(VAF_f4)
                all_VAF_mu4.append(VAF_mu4)
                "sigle plot for every simulation"
                # plt.figure('4Accumulative{}'.format(i_ra), figsize=[5, 5], linewidth=1)
                # trendline(VAF_f4, VAF_cumulative4, VAF_number4, 4, c='b', )
                # plt.scatter(VAF_f4, VAF_cumulative4, s=10)
                # plt.legend(labels='',loc='lower right', shadow=bool, title='Cancer Num= {}'.format(cancer_num4), numpoints=None)
                # plt.plot(VAF_f4, yvals4, 'r', )
                # plt.ylabel("Cumulative Number of Mutation")
                jacard_run4=jacard_run(i_cut,XY_range4, all_mutation_dict4)
            # slice 5
            if i_cut == 5:
                X_center = np.random.randint((cut_point1 + d5*2**0.5), (cut_point2 - d5*2**0.5))
                Y_center = np.random.randint((cut_point3 + d5*2**0.5), (cut_point4 - d5*2**0.5))
                cancer_num5 = np.sum(cell_matrix[X_center - d5:X_center + d5, Y_center - d5:Y_center + d5] != 0)
                while cancer_num5 != 2500:
                    X_center = np.random.randint((cut_point1 + d5 * 2 ** 0.5), (cut_point2 - d5 * 2 ** 0.5))
                    Y_center = np.random.randint((cut_point3 + d5 * 2 ** 0.5), (cut_point4 - d5 * 2 ** 0.5))
                    cancer_num5 = np.sum(
                        cell_matrix[X_center - d5:X_center + d5, Y_center - d5:Y_center + d5] != 0)
                XY_center = (X_center, Y_center)
                XY_range5.append(XY_center)

                cancer_num5 = np.sum(cell_matrix[X_center - d5:X_center + d5, Y_center - d5:Y_center + d5] != 0)

                for cut in range(0, len(cell_r)):
                    x0 = cell_r[cut]
                    y0 = cell_c[cut]

                    if X_center - d5 <= x0 < X_center + d5 and Y_center - d5 <= y0 < Y_center + d5:
                        horizon_x5.append(x0)
                        horizon_y5.append(y0)


                    mutation_ID = cell_dictionary[x0, y0].mutation  # get all the mutation ID in the dictionary
                    all_mutation_ID += mutation_ID  # Collect every mutation.ID in the all_mutation_ID
                for cut5 in range(0, len(horizon_x5)):

                    x5 = horizon_x5[cut5]
                    y5 = horizon_y5[cut5]
                    mutation_ID5 = cell_dictionary[x5, y5].mutation  # get all the mutation ID in the dictionary
                    all_mutation_ID5 += mutation_ID5  # Collect every mutation.ID in the all_mutation_ID1

                result5 = Counter(all_mutation_ID5)  # Count frequency of each mutation ID
                count_mu5 = []  # empty list: record the mutation ID in the VAF
                count_times5 = []  # empty list: record the frequency of mutation ID in the VAF
                show_ID5 = {}
                for i in result5:
                    if result5[i] >= cancer_num5 / Filter_frequency:
                        count_mu5.append(i)
                        count_times5.append(result5[i])
                    else:
                        continue
                all_mutation_dict5[XY_center] = count_mu5
                # counter the mutation proberbility in the clip area
                VAF5 = list(map(lambda c: c / (cancer_num5),
                                 count_times5))  # Object function change to list, and calculate each element of count_times

                result_VAF5 = Counter(VAF5)  # Count each element in VAF, feedback the numbers

                VAF_mu5 = []  # empty list: record the mutation ID in the VAF
                VAF_times5 = []  # empty list: record the each ID frequency in the VAF
                for i5 in result_VAF5:
                    VAF_mu5.append(i5)
                list.sort(VAF_mu5, reverse=1)
                # VAF_mu5 = VAF_mu5[0: (len(VAF_mu5) - 3)]
                # ## 2020.7.7 filter the result less than 50
                # VAF_mu5 = list(filter(lambda x: x >= Filter_frequency, VAF_mu5))
                # VAF_mu5.remove(0.5)
                for j5 in VAF_mu5:
                    VAF_times5.append(result_VAF5[j5])

                VAF_cumulative5 = []
                VAF_number5 = list(VAF_times5)

                for f5 in range(1, len(VAF_number5)+1):
                    VAF_cumulative5.append(np.sum(VAF_number5[0:f5]))
                VAF_f5 = list(map(lambda c: 1 / c, VAF_mu5))
                yvals5 = list(map(lambda c: 10 * c-20, VAF_f5))
                all_VAF_cumulative5.append(VAF_cumulative5)
                all_VAF_number5.append(VAF_number5)
                all_VAF_f5.append(VAF_f5)
                all_VAF_mu5.append(VAF_mu5)
                "sigle plot for every simulation"
                # plt.figure('5Accumulative{}'.format(i_ra), figsize=[5, 5], linewidth=1)
                # trendline(VAF_f5, VAF_cumulative5, VAF_number5, 5, c='b')
                # plt.scatter(VAF_f5, VAF_cumulative5, s=10)
                # plt.legend(labels='',loc='lower right', shadow=bool, title='Cancer Num= {}'.format(cancer_num5), numpoints=None)
                # plt.plot(VAF_f5, yvals5, 'r', )
                # plt.ylabel("Cumulative Number of Mutation")
                jacard_run5=jacard_run(i_cut,XY_range5, all_mutation_dict5)
            # slice 6
            if i_cut == 6:
                X_center = np.random.randint((cut_point1 + d6*2**0.5), (cut_point2 - d6*2**0.5))
                Y_center = np.random.randint((cut_point3 + d6*2**0.5), (cut_point4 - d6*2**0.5))
                cancer_num6 = np.sum(cell_matrix[X_center - d6:X_center + d6, Y_center - d6:Y_center + d6] != 0)
                while cancer_num6 != 3600:
                    X_center = np.random.randint((cut_point1 + d6 * 2 ** 0.5), (cut_point2 - d6 * 2 ** 0.5))
                    Y_center = np.random.randint((cut_point3 + d6 * 2 ** 0.5), (cut_point4 - d6 * 2 ** 0.5))
                    cancer_num6 = np.sum(
                        cell_matrix[X_center - d6:X_center + d6, Y_center - d6:Y_center + d6] != 0)
                XY_center = (X_center, Y_center)
                XY_range6.append(XY_center)

                cancer_num6 = np.sum(
                    cell_matrix[X_center - d6:X_center + d6, Y_center - d6:Y_center + d6] != 0)

                for cut in range(0, len(cell_r)):
                    x0 = cell_r[cut]
                    y0 = cell_c[cut]

                    if X_center - d6 <= x0 < X_center + d6 and Y_center - d6 <= y0 < Y_center + d6:
                        horizon_x6.append(x0)
                        horizon_y6.append(y0)

                    mutation_ID = cell_dictionary[x0, y0].mutation  # get all the mutation ID in the dictionary
                    all_mutation_ID += mutation_ID  # Collect every mutation.ID in the all_mutation_ID
                for cut6 in range(0, len(horizon_x6)):
                    x6 = horizon_x6[cut6]
                    y6 = horizon_y6[cut6]
                    mutation_ID6 = cell_dictionary[x6, y6].mutation  # get all the mutation ID in the dictionary

                    all_mutation_ID6 += mutation_ID6  # Collect every mutation.ID in the all_mutation_ID1

                result6 = Counter(all_mutation_ID6)  # Count frequency of each mutation ID
                count_mu6 = []  # empty list: record the mutation ID in the VAF
                count_times6 = []  # empty list: record the frequency of mutation ID in the VAF

                for i in result6:
                    if result6[i] >= cancer_num6 / Filter_frequency:
                        count_mu6.append(i)
                        count_times6.append(result6[i])
                    else:
                        continue
                all_mutation_dict6[XY_center] = count_mu6
                # counter the mutation proberbility in the clip area
                VAF6 = list(map(lambda c: c / (cancer_num6),
                                 count_times6))  # Object function change to list, and calculate each element of count_times

                result_VAF6 = Counter(VAF6)  # Count each element in VAF, feedback the numbers

                VAF_mu6 = []  # empty list: record the mutation ID in the VAF
                VAF_times6 = []  # empty list: record the each ID frequency in the VAF
                for i6 in result_VAF6:
                    VAF_mu6.append(i6)
                list.sort(VAF_mu6, reverse=1)
                # VAF_mu6 = VAF_mu6[0: (len(VAF_mu6) - 3)]
                # ## 2020.7.7 filter the result less than 50
                # VAF_mu6 = list(filter(lambda x: x >= Filter_frequency, VAF_mu6))
                # VAF_mu6.remove(0.5)
                for j6 in VAF_mu6:
                    VAF_times6.append(result_VAF6[j6])

                VAF_cumulative6 = []
                VAF_number6 = list(VAF_times6)
                for f6 in range(1,1+len(VAF_number6)):
                    VAF_cumulative6.append(np.sum(VAF_number6[0:f6]))
                VAF_f6 = list(map(lambda c: 1 / c, VAF_mu6))
                yvals6  = list(map(lambda c: 10 * c-20, VAF_f6))

                all_VAF_f6.append(VAF_f6)
                all_VAF_number6.append(VAF_number6)
                all_VAF_cumulative6.append(VAF_cumulative6)
                all_VAF_mu6.append(VAF_mu6)
                "sigle plot for every simulation"
                # plt.figure('6Accumulative{}'.format(i_ra), figsize=[5, 5], linewidth=1)
                # trendline(VAF_f6, VAF_cumulative6, VAF_number6, 6, c='b')
                # plt.scatter(VAF_f6, VAF_cumulative6, s=10)
                # plt.legend(labels='',loc='lower right', shadow=bool, title='Cancer Num= {}'.format(cancer_num6), numpoints=None)
                # plt.plot(VAF_f6, yvals6, 'r', )
                # plt.ylabel("Cumulative Number of Mutation")
                jacard_run6 = jacard_run(i_cut,XY_range6, all_mutation_dict6)
    combine_set = combine(XY_range1,2)
    X_distance = []
    Y_jaccard = []
    for i, j in combine_set:
        i_x,i_y = i
        j_x, j_y = j
        D_i_j = math.sqrt((j_x-i_x)**2+(j_y-i_y)**2)   # calculate the distance of two combination points
        X_distance.append(D_i_j)
        Y_jaccard.append(jacard_index(all_mutation_dict1[i],all_mutation_dict1[j]))

    # #     print('assssss',i, j,D_i_j,jacard_index(all_mutation_dict1[i],all_mutation_dict3[j]), )
    # # print('zzz',math.factorial(10),len(X_distance),len(Y_jaccard))
    # #
    plt.figure('jacard_index all', figsize=[5, 5])
    plt.scatter(X_distance, Y_jaccard, c='#65CE6C',s=10)
    plt.xlabel('Distance')
    plt.ylabel("Jaccard index")
    plt.legend(labels='',loc='upper right', shadow=bool, title='Jaccard Num= {}'.format(len(Y_jaccard)), numpoints=None)


    print('aa',jacard_run1[0],jacard_run1[1],'bb',jacard_run2[0],jacard_run2[1])
    a1,b1= jacard_run1
    a2, b2 = jacard_run2
    a3, b3 = jacard_run3

    Jacard_boxplot(1,b1,a1)
    Jacard_boxplot(2,b2,a2)
    Jacard_boxplot(3,b3,a3)
    Jacard_boxplot(4,jacard_run4[1],jacard_run4[0])
    Jacard_boxplot(5,jacard_run5[1],jacard_run5[0])
    Jacard_boxplot(6,jacard_run6[1],jacard_run6[0])

    end_time = time.time()
    run_time = end_time - start_time

    print("time:",run_time)



    plt.figure(('whole VAF all'), figsize=[12,8], linewidth=1)
    plt.subplot(231)
    for i, j, z in zip(all_VAF_mu1, all_VAF_cumulative1, all_VAF_number1):
        plt.bar(i, z, width=0.005, color='seagreen',alpha=0.5,)

    plt.legend(labels='',loc='upper right', shadow=bool, title='Cancer Num= {}'.format(cancer_num1), numpoints=None)
    plt.ylabel("Number of Mutation")
    plt.title('Whole VAF')
    plt.subplot(232)
    for i, j, z in zip(all_VAF_mu2, all_VAF_cumulative2, all_VAF_number2):
        plt.bar(i, z, width=0.005, color='seagreen',alpha=0.5,)
    plt.legend(labels='',loc='upper right', shadow=bool, title='Cancer Num= {}'.format(cancer_num2), numpoints=None,
               )
    plt.subplot(233)
    for i, j, z in zip(all_VAF_mu3, all_VAF_cumulative3, all_VAF_number3):
        plt.bar(i, z, width=0.005, color='seagreen',alpha=0.5,)

    plt.legend(labels='',loc='upper right', shadow=bool, title='Cancer Num= {}'.format(cancer_num3), numpoints=None)
    plt.subplot(234)
    for i, j, z in zip(all_VAF_mu4, all_VAF_cumulative4, all_VAF_number4):
        plt.bar(i, z,width=0.005, color='seagreen',alpha=0.5)

    plt.legend(labels='',loc='upper right', shadow=bool, title='Cancer Num= {}'.format(cancer_num4), numpoints=None)
    plt.ylabel("Number of Mutation")
    plt.xlabel('Variant allelic frequency')
    plt.subplot(235)
    for i, j, z in zip(all_VAF_mu5, all_VAF_cumulative5, all_VAF_number5):
        plt.bar(i, z,width=0.005, color='seagreen',alpha=0.5)
    plt.legend(labels='',loc='upper right', shadow=bool, title='Cancer Num= {}'.format(cancer_num5), numpoints=None)
    plt.xlabel('Variant allelic frequency')
    plt.subplot(236)
    for i, j, z in zip(all_VAF_mu6, all_VAF_cumulative6, all_VAF_number6):
        plt.bar(i, z, width=0.005, color='seagreen',alpha=0.5,)
    plt.legend(labels='',loc='upper right', shadow=bool, title='Cancer Num= {}'.format(cancer_num6), numpoints=None)
    plt.xlabel('Variant allelic frequency')






    plt.figure(('whole accumulative all'), figsize=[12,8], linewidth=1)
    plt.subplot(231)
    for i, j, z in zip(all_VAF_f1, all_VAF_cumulative1, all_VAF_number1):
        trendline(i, j, z, 1,  c='b', alpha=0.2, Rval=True)
        plt.scatter(i, j, s=10)

    plt.legend(labels='',loc='upper left', shadow=bool, title='Cancer Num= {}'.format(cancer_num1), numpoints=None)
    plt.plot(VAF_f1, yvals1, 'r', )
    plt.ylabel("Cumulative Number of Mutation")
    plt.title('Whole times Accumulative')
    plt.subplot(232)
    plt.plot(VAF_f2, yvals2, 'r', )
    for i, j, z in zip(all_VAF_f2, all_VAF_cumulative2, all_VAF_number2):
        trendline(i, j, z, 2, c='r', Rval=True)
        plt.scatter(i, j, s=10)
    plt.legend(labels='',loc='upper left', shadow=bool, title='Cancer Num= {}'.format(cancer_num2), numpoints=None,
               )
    plt.subplot(233)
    plt.plot(VAF_f3, yvals3, 'r', )
    for i, j, z in zip(all_VAF_f3, all_VAF_cumulative3, all_VAF_number3):
        trendline(i, j, z, 3, c='y', Rval=True)
        plt.scatter(i, j, s=10)

    plt.legend(labels='',loc='upper left', shadow=bool, title='Cancer Num= {}'.format(cancer_num3), numpoints=None)
    plt.subplot(234)
    plt.plot(VAF_f4, yvals4, 'r', )
    for i, j, z in zip(all_VAF_f4, all_VAF_cumulative4, all_VAF_number4):
        trendline(i, j, z, 4, Rval=True)
        plt.scatter(i, j, s=10)

    plt.legend(labels='',loc='upper left', shadow=bool, title='Cancer Num= {}'.format(cancer_num4), numpoints=None)
    plt.ylabel("Cumulative Number of Mutation")
    plt.xlabel('Inverse allelic frequency 1/f')
    plt.subplot(235)
    plt.plot(VAF_f5, yvals5, 'r', )
    for i, j, z in zip(all_VAF_f5, all_VAF_cumulative5, all_VAF_number5):
        trendline(i, j, z, 5, c='g', Rval=True)
        plt.scatter(i, j, s=10)
    plt.legend(labels='',loc='upper left', shadow=bool, title='Cancer Num= {}'.format(cancer_num5), numpoints=None)
    plt.xlabel('Inverse allelic frequency 1/f')
    plt.subplot(236)
    plt.plot(VAF_f6, yvals6, 'r', )
    for i, j, z in zip(all_VAF_f6, all_VAF_cumulative6, all_VAF_number6):
        trendline(i, j, z, 6, c='tan', Rval=True)
        plt.scatter(i, j, s=10)
    plt.legend(labels='',loc='upper left', shadow=bool, title='Cancer Num= {}'.format(cancer_num6), numpoints=None)
    plt.xlabel('Inverse allelic frequency 1/f')

    plt.figure('Whole VAF',figsize = [5,5])
    plt.bar(VAF_mu, VAF_times, width=0.005, color='seagreen')
    plt.xlabel('VAF')
    plt.ylim(0,200)
    plt.ylabel("Number of Mutation")
    plt.legend(labels='',loc='upper right', shadow=bool, title='Cancer Num= {}'.format(cancer_num), numpoints=None)

    plt.figure('Whole cumulative',figsize = [5,5])
    plt.scatter(VAF_f, VAF_cumulative,c='k')
    D_ks,W_KS=Distance_KS(VAF_f, VAF_cumulative)
    plt.plot(VAF_f, yvals, 'r', )
    plt.xlabel('Inverse allelic frequency 1/f')
    plt.ylabel("Cumulative Number of Mutations M(f)")
    plt.legend(labels='',loc='lower right', shadow=bool, title='Cancer Num= {}'.format(cancer_num), numpoints=None)
    plt.scatter(VAF_f, VAF_cumulative )
    print('Whole_KS',D_ks,W_KS,'  Cancer number:',cancer_num)
    map = matplotlib.colors.ListedColormap(
        ['#FFFFFF', '#BA4EF9', '#BA4EF9', '#E8E20A', '#EF6273',
         '#EFABAB', ])
    plt.figure('cancer growth',figsize = [7.5,6])
    plt.title('Cancer Number is: {}'.format(cancer_num))
    sns.heatmap(cancer_matrix,cmap='CMRmap_r')
    plt.xticks([])
    plt.yticks([])
    plt.figure('Cell growth',figsize = [5,5])
    plt.title('Cancer Number is: {}'.format(cancer_num))
    for i, j in XY_range1:
        plt.text(i, j, '①', color='gold', )
    for i, j in XY_range2:
        plt.text(i, j, '②', color='k')
    for i, j in XY_range3:
        plt.text(i, j, '③', color='g')
    for i, j in XY_range4:
        plt.text(i, j, '④', color='b')
    for i, j in XY_range5:
        plt.text(i, j, '⑤', color='m')
    for i, j in XY_range6:
        plt.text(i, j, '⑥', color='c')

    plt.imshow(cell_matrix, cmap=map)

    pdf = matplotlib.backends.backend_pdf.PdfPages('{} %s.pdf'.format(push_rate) % (random.randint(1,100)))
    for fig in range(1, 32):  ## will open an empty extra figure :(
        pdf.savefig(fig)

    pdf.close()
    plt.show()
    plt.ion()




