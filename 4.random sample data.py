'''
mistake: 1.growth disperse
        2. do not update with death
        3. mutation_ID = [0] repeat with origin mutation_ID
  [Cancer growth model]
  [2021/6/28]

  9 Area  10*10      2^15 cancer cell   death rate 0.5   mutation rate =10
Plot vaf   cumulative      fitness cumulative

 The code using the Object functions and structs to define the ID
 and mutation type of daughter cell, and the overall code
 is shorter and several times faster, in addition, the test
 code could detect the whole cancer cell VAF.
'''

import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors   # define the colour by myself
import pandas as pd    # Applied to cell pushing
import time     # calculate the spend time of code
from collections import Counter     # count the frequency of mutation{{}
import matplotlib.backends.backend_pdf
start_time = time.time()
Max_generation = 13
push_rate = 1
Max_ROW = 700
Max_COL = 700

### Basic parameter setting
background_gene = 1   # The background gene that cancer originally carried
mutation_rate = 0.0
birth_rate = 1
death_rate = 0.0
# birth_rate = int(input('Birth rate ='))
# death_rate = int(input('death rate ='))  # The death probability of cell growth process
die_divide_times = 500   # The max times of cell divides,  arrived this point the cell will die
Poisson_lambda = 10   # the mean value of Poisson distributions


mutation_type2 = 1
mutation_type3 = 20
mutation_type4 = 30
mutation_type5 = 40

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

def trendline(xd, yd,term, order=1, c='r', alpha=0.1, Rval=False):
    """Make a line of best fit"""

    #Calculate trendline
    coeffs = np.polyfit(xd, yd, order)

    intercept = coeffs[-1]
    slope = coeffs[-2]
    power = coeffs[0] if order == 2 else 0

    minxd = np.min(xd)
    maxxd = np.max(xd)

    xl = np.array([minxd, maxxd])
    yl = power * xl ** 2 + slope * xl + intercept

    #Plot trendline
    plt.plot(xl, yl,'b', alpha=alpha)

    #Calculate R Squared
    p = np.poly1d(coeffs)

    ybar = np.sum(yd) / len(yd)
    ssreg = np.sum((p(xd) - ybar) ** 2)
    sstot = np.sum((yd - ybar) ** 2)
    Rsqr = ssreg / sstot

    if not Rval:
        #Plot R^2 value
        plt.text(0.5 * maxxd + 0.2 * minxd, 0.9 * np.max(yd) + 0.2 * np.min(yd),
                 '$R^2 = %0.3f$' % Rsqr)
        plt.text(0.5 * maxxd + 0.2 * minxd, 0.8 * np.max(yd) + 0.2 * np.min(yd),
                 "S: %.2f" % slope )
    else:
        #Return the R^2 value:
        print('R{}'.format(term), Rsqr)

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
            diagnal_weight = len(diagnal_empty) / (len(diagnal_empty) +   len(square_empty))
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

    XY_range =[]
    all_VAF_f11 =[]

    all_VAF_f12 = []
    all_VAF_f13 = []
    all_VAF_f14 = []
    all_VAF_f15 = []
    all_VAF_f16 = []

    all_VAF_cumulative11 = []
    all_VAF_cumulative12 = []
    all_VAF_cumulative13 = []
    all_VAF_cumulative14 = []
    all_VAF_cumulative15 = []
    all_VAF_cumulative16 = []

    for i_ra in range(1,41):
        X_circle = (cut_point1+45), (cut_point2-45)
        Y_circle = (cut_point3+45), (cut_point4-45)

        X_center = np.random.randint((cut_point1+45), (cut_point2-45))
        Y_center = np.random.randint((cut_point3+45), (cut_point4-45))
        XY_center = (X_center,Y_center)
        XY_range.append(XY_center)
        distance = 5
        d11 = distance
        d12 = d11+distance
        d13 = d12+distance
        d14 = d13+distance
        d15 = d14+distance
        d16 = d15+distance

        # cut the tissue into 5 piece

        cancer_num11 = np.sum(cell_matrix[X_center-d11:X_center+d11, Y_center-d11:Y_center+d11] != 0)
        cancer_num12 = np.sum(cell_matrix[X_center-d12:X_center+d12, Y_center-d12:Y_center+d12] != 0)
        cancer_num13 = np.sum(cell_matrix[X_center-d13:X_center+d13, Y_center-d13:Y_center+d13] != 0)
        cancer_num14 = np.sum(cell_matrix[X_center-d14:X_center+d14, Y_center-d14:Y_center+d14] != 0)
        cancer_num15 = np.sum(cell_matrix[X_center-d15:X_center+d15, Y_center-d15:Y_center+d15] != 0)
        cancer_num16 = np.sum(cell_matrix[X_center-d16:X_center+d16, Y_center-d16:Y_center+d16] != 0)

        # cancer_num17 = np.sum(cell_matrix[X_center-d1:X_center+d1, 15:255] != 0)
        # cancer_num18 = np.sum(cell_matrix[X_center-d1:175, 15:255] != 0)
        # cancer_num19 = np.sum(cell_matrix[X_center-d1:180, 15:255] != 0)

        # print('ccc3',' 1-9', cancer_num11, cancer_num12, cancer_num13, cancer_num14, cancer_num15,cancer_num16)

        ratio1 = mutation_num2/cancer_num
        ratio2 = mutation_num3 / cancer_num
        ratio3 = mutation_num4 / cancer_num
        ratio4 = mutation_num5 / cancer_num
        # print('cancer number',cancer_num,'\n','mutation ratio2:',ratio1,'\n','mutation ratio3:',ratio2,'\n','mutation ratio4:',ratio3,'\n','mutation ratio5:',ratio4)

        end_time = time.time()
        run_time = end_time - start_time
        end_time = time.time()
        run_time = end_time - start_time

        #VAF = [(count_times_element - 1) for count_times_element in range(count_times)] #variant allele frequence

        all_mutation_ID = []
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

        for cut in range(0, len(cell_r)):
            x0 = cell_r[cut]
            y0 = cell_c[cut]


            if X_center-d11 <= x0 < X_center+d11 and Y_center-d11 <= y0 < Y_center+d11:
                horizon_x11.append(x0)
                horizon_y11.append(y0)
            if X_center-d12 <= x0 < X_center+d12 and Y_center-d12 <= y0 < Y_center+d12:
                horizon_x12.append(x0)
                horizon_y12.append(y0)
            if X_center-d13 <= x0 < X_center+d13 and Y_center-d13 <= y0 < Y_center+d13:
                horizon_x13.append(x0)
                horizon_y13.append(y0)
            if X_center-d14 <= x0 < X_center+d14 and Y_center-d14 <= y0 < Y_center+d14:
                horizon_x14.append(x0)
                horizon_y14.append(y0)
            if X_center-d15 <= x0 < X_center+d15 and Y_center-d15 <= y0 < Y_center+d15:
                horizon_x15.append(x0)
                horizon_y15.append(y0)
            if X_center-d16 <= x0 < X_center+d16 and Y_center-d16 <= y0 < Y_center+d16:
                horizon_x16.append(x0)
                horizon_y16.append(y0)
            # if X_center-d1 <= x0 < 170 and 15 <= y0 < 255:
            #     horizon_x17.append(x0)
            #     horizon_y17.append(y0)
            # if 125 <= x0 < 175 and 15 <= y0 < 255:
            #     horizon_x18.append(x0)
            #     horizon_y18.append(y0)
            # if 120 <= x0 < 180 and 15 <= y0 < 255:
            #     horizon_x19.append(x0)
            #     horizon_y19.append(y0)


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
        VAF = list(map(lambda n: n / (cancer_num),count_times))  # Object function change to list, and calculate each element of count_times
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
        for f in range(1, len(VAF_number)+1):
            VAF_cumulative.append(np.sum(VAF_number[0:f]))
            VAF_f = list(map(lambda c: 1 / c, VAF_mu))

        f = np.polyfit(VAF_f, VAF_cumulative,2)
        yvals=np.polyval(f, VAF_f)


        ### Punch tissue
        ### 11 slice
        for cut11 in range(0, len(horizon_x11)):
            x11 = horizon_x11[cut11]
            y11 = horizon_y11[cut11]
            mutation_ID11 = cell_dictionary[x11, y11].mutation  # get all the mutation ID in the dictionary

            all_mutation_ID11 += mutation_ID11  # Collect every mutation.ID in the all_mutation_ID1

        result11 = Counter(all_mutation_ID11)  # Count frequency of each mutation ID
        count_mu11 = []  # empty list: record the mutation ID in the VAF
        count_times11 = []  # empty list: record the frequency of mutation ID in the VAF
        show_ID11 = {}
        ID_key =[]

        for i in result11:
            count_mu11.append(i)
            count_times11.append(result11[i])

            show_ID11[result11[i]] = i

        # counter the mutation proberbility in the clip area
        VAF11 = list(map(lambda c: c / (  cancer_num11),
                         count_times11))  # Object function change to list, and calculate each element of count_times

        result_VAF11 = Counter(VAF11)  # Count each element in VAF, feedback the numbers

        VAF_mu11 = []  # empty list: record the mutation ID in the VAF
        VAF_times11 = []  # empty list: record the each ID frequency in the VAF
        for i11 in result_VAF11:
            VAF_mu11.append(i11)
        list.sort(VAF_mu11, reverse=1)
        for j11 in VAF_mu11:
            VAF_times11.append(result_VAF11[j11])

        VAF_cumulative11 = []
        VAF_number11 = list(VAF_times11)
        for f11 in range(1, 1+len(VAF_number11)):
            VAF_cumulative11.append(np.sum(VAF_number11[0:f11]))
        VAF_f11 = list(map(lambda c: 1 / c, VAF_mu11))
        yvals11 = list(map(lambda c: 10 * c, VAF_f11))

        ### 12 slice
        for cut12 in range(0, len(horizon_x12)):
            x12 = horizon_x12[cut12]
            y12 = horizon_y12[cut12]
            mutation_ID12 = cell_dictionary[x12, y12].mutation  # get all the mutation ID in the dictionary

            all_mutation_ID12 += mutation_ID12  # Collect every mutation.ID in the all_mutation_ID1

        result12 = Counter(all_mutation_ID12)  # Count frequency of each mutation ID
        count_mu12 = []  # empty list: record the mutation ID in the VAF
        count_times12 = []  # empty list: record the frequency of mutation ID in the VAF
        show_ID12={}
        for i in result12:
            count_mu12.append(i)
            count_times12.append(result12[i])
            show_ID12[result12[i]] = i
        # counter the mutation proberbility in the clip area
        VAF12 = list(map(lambda c: c / (  cancer_num12),
                         count_times12))  # Object function change to list, and calculate each element of count_times

        result_VAF12 = Counter(VAF12)  # Count each element in VAF, feedback the numbers

        VAF_mu12 = []  # empty list: record the mutation ID in the VAF
        VAF_times12 = []  # empty list: record the each ID frequency in the VAF
        for i12 in result_VAF12:
            VAF_mu12.append(i12)
        list.sort(VAF_mu12, reverse=1)
        for j12 in VAF_mu12:
            VAF_times12.append(result_VAF12[j12])

        VAF_cumulative12 = []
        VAF_number12 = list(VAF_times12)
        for f12 in range(1,1+len(VAF_number12)):
            VAF_cumulative12.append(np.sum(VAF_number12[0:f12]))
        VAF_f12 = list(map(lambda c: 1 / c, VAF_mu12))
        yvals12 = list(map(lambda c: 10 * c, VAF_f12))
        ## 13 slice
        for cut13 in range(0, len(horizon_x13)):
            x13 = horizon_x13[cut13]
            y13 = horizon_y13[cut13]
            mutation_ID13 = cell_dictionary[x13, y13].mutation  # get all the mutation ID in the dictionary

            all_mutation_ID13 += mutation_ID13  # Collect every mutation.ID in the all_mutation_ID1

        result13 = Counter(all_mutation_ID13)  # Count frequency of each mutation ID
        count_mu13 = []  # empty list: record the mutation ID in the VAF
        count_times13 = []  # empty list: record the frequency of mutation ID in the VAF
        show_ID13 = {}
        for i in result13:
            count_mu13.append(i)
            show_ID13[result13[i]] = i
            count_times13.append(result13[i])

        # counter the mutation proberbility in the clip area
        VAF13 = list(map(lambda c: c / (  cancer_num13),
                         count_times13))  # Object function change to list, and calculate each element of count_times

        result_VAF13 = Counter(VAF13)  # Count each element in VAF, feedback the numbers

        VAF_mu13 = []  # empty list: record the mutation ID in the VAF
        VAF_times13 = []  # empty list: record the each ID frequency in the VAF
        for i13 in result_VAF13:
            VAF_mu13.append(i13)
        list.sort(VAF_mu13, reverse=1)
        for j13 in VAF_mu13:
            VAF_times13.append(result_VAF13[j13])

        VAF_cumulative13 = []
        VAF_number13 = list(VAF_times13)
        for f13 in range(1,1+len(VAF_number13)):
            VAF_cumulative13.append(np.sum(VAF_number13[0:f13]))
        VAF_f13 = list(map(lambda c: 1 / c, VAF_mu13))
        yvals13 = list(map(lambda c: 10 * c, VAF_f13))
        # 14 slice
        for cut14 in range(0, len(horizon_x14)):
            x14 = horizon_x14[cut14]
            y14 = horizon_y14[cut14]
            mutation_ID14 = cell_dictionary[x14, y14].mutation  # get all the mutation ID in the dictionary

            all_mutation_ID14 += mutation_ID14  # Collect every mutation.ID in the all_mutation_ID1

        result14 = Counter(all_mutation_ID14)  # Count frequency of each mutation ID
        count_mu14 = []  # empty list: record the mutation ID in the VAF
        count_times14 = []  # empty list: record the frequency of mutation ID in the VAF
        show_ID14 = {}
        for i in result14:
            count_mu14.append(i)
            count_times14.append(result14[i])
            show_ID14[result14[i]] = i

        # counter the mutation proberbility in the clip area
        VAF14 = list(map(lambda c: c / (  cancer_num14),
                         count_times14))  # Object function change to list, and calculate each element of count_times

        result_VAF14 = Counter(VAF14)  # Count each element in VAF, feedback the numbers

        VAF_mu14 = []  # empty list: record the mutation ID in the VAF
        VAF_times14 = []  # empty list: record the each ID frequency in the VAF
        for i14 in result_VAF14:
            VAF_mu14.append(i14)
        list.sort(VAF_mu14, reverse=1)
        for j14 in VAF_mu14:
            VAF_times14.append(result_VAF14[j14])

        VAF_cumulative14 = []
        VAF_number14 = list(VAF_times14)
        for f14 in range(1,1+len(VAF_number14)):
            VAF_cumulative14.append(np.sum(VAF_number14[0:f14]))
        VAF_f14 = list(map(lambda c: 1 / c, VAF_mu14))
        yvals14 = list(map(lambda c: 10 * c, VAF_f14))

        # slice 15
        for cut15 in range(0, len(horizon_x15)):

            x15 = horizon_x15[cut15]
            y15 = horizon_y15[cut15]
            mutation_ID15 = cell_dictionary[x15, y15].mutation  # get all the mutation ID in the dictionary
            all_mutation_ID15 += mutation_ID15  # Collect every mutation.ID in the all_mutation_ID1

        result15 = Counter(all_mutation_ID15)  # Count frequency of each mutation ID
        count_mu15 = []  # empty list: record the mutation ID in the VAF
        count_times15 = []  # empty list: record the frequency of mutation ID in the VAF
        show_ID15 = {}
        for i in result15:
            count_mu15.append(i)
            count_times15.append(result15[i])
            show_ID15[result15[i]] = i

        # counter the mutation proberbility in the clip area
        VAF15 = list(map(lambda c: c / (  cancer_num15),
                         count_times15))  # Object function change to list, and calculate each element of count_times

        result_VAF15 = Counter(VAF15)  # Count each element in VAF, feedback the numbers

        VAF_mu15 = []  # empty list: record the mutation ID in the VAF
        VAF_times15 = []  # empty list: record the each ID frequency in the VAF
        for i15 in result_VAF15:
            VAF_mu15.append(i15)
        list.sort(VAF_mu15, reverse=1)
        for j15 in VAF_mu15:
            VAF_times15.append(result_VAF15[j15])

        VAF_cumulative15 = []
        VAF_number15 = list(VAF_times15)

        for f15 in range(1, len(VAF_number15)+1):
            VAF_cumulative15.append(np.sum(VAF_number15[0:f15]))
        VAF_f15 = list(map(lambda c: 1 / c, VAF_mu15))
        yvals15 = list(map(lambda c: 10 * c, VAF_f15))
        # slice 16
        for cut16 in range(0, len(horizon_x16)):
            x16 = horizon_x16[cut16]
            y16 = horizon_y16[cut16]
            mutation_ID16 = cell_dictionary[x16, y16].mutation  # get all the mutation ID in the dictionary

            all_mutation_ID16 += mutation_ID16  # Collect every mutation.ID in the all_mutation_ID1

        result16 = Counter(all_mutation_ID16)  # Count frequency of each mutation ID
        count_mu16 = []  # empty list: record the mutation ID in the VAF
        count_times16 = []  # empty list: record the frequency of mutation ID in the VAF
        show_ID16 = {}
        for i in result16:
            count_mu16.append(i)
            count_times16.append(result16[i])
            show_ID16[result16[i]] = i
        # counter the mutation proberbility in the clip area
        VAF16 = list(map(lambda c: c / (  cancer_num16),
                         count_times16))  # Object function change to list, and calculate each element of count_times

        result_VAF16 = Counter(VAF16)  # Count each element in VAF, feedback the numbers

        VAF_mu16 = []  # empty list: record the mutation ID in the VAF
        VAF_times16 = []  # empty list: record the each ID frequency in the VAF
        for i16 in result_VAF16:
            VAF_mu16.append(i16)
        list.sort(VAF_mu16, reverse=1)
        for j16 in VAF_mu16:
            VAF_times16.append(result_VAF16[j16])

        VAF_cumulative16 = []
        VAF_number16 = list(VAF_times16)
        for f16 in range(1,1+len(VAF_number16)):
            VAF_cumulative16.append(np.sum(VAF_number16[0:f16]))
        VAF_f16 = list(map(lambda c: 1 / c, VAF_mu16))
        yvals16  = list(map(lambda c: 10 * c, VAF_f16))

        all_VAF_f11.append(VAF_f11)
        all_VAF_f12.append(VAF_f12)
        all_VAF_f13.append(VAF_f13)
        all_VAF_f14.append(VAF_f14)
        all_VAF_f15.append(VAF_f15)
        all_VAF_f16.append(VAF_f16)

        all_VAF_cumulative11.append(VAF_cumulative11)
        all_VAF_cumulative12.append(VAF_cumulative12)
        all_VAF_cumulative13.append(VAF_cumulative13)
        all_VAF_cumulative14.append(VAF_cumulative14)
        all_VAF_cumulative15.append(VAF_cumulative15)
        all_VAF_cumulative16.append(VAF_cumulative16)


        print(i_ra,'11', VAF_f11, '\n', i_ra,'11', VAF_cumulative11)
        print(i_ra,'12', VAF_f12, '\n', i_ra,'12', VAF_cumulative12)
        print(i_ra,'13', VAF_f13, '\n', i_ra,'13', VAF_cumulative13)
        print(i_ra,'14', VAF_f14, '\n', i_ra,'14', VAF_cumulative14)
        print(i_ra,'15', VAF_f15, '\n', i_ra,'15', VAF_cumulative15)
        print(i_ra,'16', VAF_f16, '\n', i_ra,'16', VAF_cumulative16)

        print(i_ra,'11', VAF_mu11, '\n', i_ra,'11', VAF_times11)
        print(i_ra,'12', VAF_mu12, '\n', i_ra,'12', VAF_times12)
        print(i_ra,'13', VAF_mu13, '\n', i_ra,'13', VAF_times13)
        print(i_ra,'14', VAF_mu14, '\n', i_ra,'14', VAF_times14)
        print(i_ra,'15', VAF_mu15, '\n', i_ra,'15', VAF_times15)
        print(i_ra,'16', VAF_mu16, '\n', i_ra,'16', VAF_times16)

        plt.figure('Horizon VAF', figsize=[12, 8], linewidth=5)

        # plt.bar(VAF_mu, VAF_times, width=0.01, color='seagreen',alpha = 0.3)
        # plt.xlabel("VAF")
        # plt.ylim(0,200)
        # plt.ylabel("Number of Mutation")
        plt.subplot(231)

        plt.bar(VAF_mu11, VAF_times11, width=0.01, color='seagreen',alpha = 0.05)
        plt.xlabel("Variant Allele Frequency(A)")
        plt.ylim(0, 200)
        plt.legend(loc='upper right', shadow=bool, title='Cancer Num= {}'.format(cancer_num11), numpoints=None)
        plt.ylabel("Number of Mutation")
        plt.subplot(232)

        plt.bar(VAF_mu12, VAF_times12, width=0.01, color='seagreen',alpha = 0.05)
        plt.xlabel("Variant Allele Frequency(B)")
        plt.ylim(0, 200)
        plt.legend(loc='upper right', shadow=bool, title='Cancer Num= {}'.format(cancer_num12), numpoints=None)

        plt.subplot(233)
        plt.bar(VAF_mu13, VAF_times13, width=0.01, color='seagreen',alpha = 0.05)
        plt.xlabel('Variant Allele Frequency(C)')
        plt.legend(loc='upper right', shadow=bool, title='Cancer Num= {}'.format(cancer_num13), numpoints=None)
        plt.ylim(0, 200)
        plt.subplot(234)
        plt.bar(VAF_mu14, VAF_times14, width=0.01, color='seagreen',alpha = 0.05)
        plt.xlabel('Variant Allele Frequency(D)')
        plt.legend(loc='upper right', shadow=bool, title='Cancer Num= {}'.format(cancer_num14), numpoints=None)
        plt.ylabel("Number of Mutation")

        plt.ylim(0, 200)
        plt.subplot(235)
        plt.bar(VAF_mu15, VAF_times15, width=0.01, color='seagreen',alpha = 0.05)
        plt.xlabel('Variant Allele Frequency(E)')
        plt.ylim(0, 200)

        plt.legend(loc='upper right', shadow=bool, title='Cancer Num= {}'.format(cancer_num15), numpoints=None)
        plt.subplot(236)
        plt.bar(VAF_mu16, VAF_times16, width=0.01, color='seagreen',alpha = 0.05)
        plt.xlabel("Variant Allele Frequency(F)")

        plt.ylim(0, 200)
        plt.legend(loc='upper right', shadow=bool, title='Cancer Num= {}'.format(cancer_num16), numpoints=None)

    plt.figure(('whole accumulative all'), figsize=[12, 8], linewidth=1)
    plt.subplot(231)
    for i,j in zip(all_VAF_f11, all_VAF_cumulative11) :

        trendline(i, j,11,Rval= True,alpha=0.5)
        plt.scatter(i, j, s=10)

    plt.legend(loc='lower right', shadow=bool, title='Cancer Num= {}'.format(cancer_num11), numpoints=None)
    plt.plot(VAF_f11, yvals11, 'r', )
    plt.ylabel("Cumulative Number of Mutation")
    plt.subplot(232)
    # plt.title('Whole times Accumulative')
    plt.plot(VAF_f12, yvals12, 'r', )
    for i, j in zip(all_VAF_f12, all_VAF_cumulative12):
        trendline(i, j,12,Rval= True,)
        plt.scatter(i, j, s=10)
    plt.legend(loc='lower right', shadow=bool, title='Cancer Num= {}'.format(cancer_num12), numpoints=None,
               )
    plt.subplot(233)
    plt.plot(VAF_f13, yvals13, 'r', )
    for i, j in zip(all_VAF_f13, all_VAF_cumulative13):
        trendline(i, j,13,Rval= True)
        plt.scatter(i, j, s=10)

    plt.legend(loc='lower right', shadow=bool, title='Cancer Num= {}'.format(cancer_num13), numpoints=None)
    plt.subplot(234)
    plt.plot(VAF_f14, yvals14, 'r', )
    for i, j in zip(all_VAF_f14, all_VAF_cumulative14):
        trendline(i, j,14,Rval= True)
        plt.scatter(i, j, s=10)
    plt.legend(loc='lower right', shadow=bool, title='Cancer Num= {}'.format(cancer_num14), numpoints=None)
    plt.ylabel("Cumulative Number of Mutation")
    plt.xlabel('Inverse allelic frequency 1/f')
    plt.subplot(235)
    plt.plot(VAF_f15, yvals15, 'r', )
    for i, j in zip(all_VAF_f15, all_VAF_cumulative15):
        trendline(i, j,15,Rval= True)
        plt.scatter(i, j, s=10)
    plt.legend(loc=4, shadow=bool, title='Cancer Num= {}'.format(cancer_num15), numpoints=None)
    plt.xlabel('Inverse allelic frequency 1/f')
    plt.subplot(236)
    plt.plot(VAF_f16, yvals16, 'r', )
    for i, j in zip(all_VAF_f16, all_VAF_cumulative16):
        trendline(i, j,16,Rval= True)
        plt.scatter(i, j, s=10)
    plt.legend(loc=4, shadow=bool, title='Cancer Num= {}'.format(cancer_num16), numpoints=None)
    plt.xlabel('Inverse allelic frequency 1/f')

    plt.figure('Whole plot',figsize = [5,5])
    plt.bar(VAF_mu, VAF_times, width=0.01, color='seagreen',linewidth=10.0)
    plt.xlabel('VAF')
    plt.ylim(0,200)
    plt.ylabel("Number of Mutation")
    plt.legend(loc='upper right', shadow=bool, title='Cancer Num= {}'.format(cancer_num), numpoints=None)

    plt.figure('cumulative',figsize = [5,5])

    plt.scatter(VAF_f, VAF_cumulative,c='k')
    plt.plot(VAF_f, yvals, 'r', )
    plt.xlabel('Inverse allelic frequency 1/f')
    plt.ylabel("Cumulative Number of Mutations M(f)")
    plt.legend(loc='lower right', shadow=bool, title='Cancer Num= {}'.format(cancer_num), numpoints=None)
    plt.scatter(VAF_f, VAF_cumulative)
    map = matplotlib.colors.ListedColormap(
        ['#FFFFFF', '#BA4EF9', '#BA4EF9', '#E8E20A', '#EF6273',
         '#EFABAB', ])
    # plt.figure('cancer growth')
    # plt.title('Cancer Number is: {}'.format(cancer_num))
    # plt.imshow(cancer_matrix, cmap=map)
    plt.figure('Cell growth')
    plt.title('Cancer Number is: {}'.format(cancer_num))
    for i,j in XY_range:
        plt.text(i,j,'o',color='r')
    plt.imshow(cell_matrix, cmap=map)
    plt.show()
    plt.ion()




