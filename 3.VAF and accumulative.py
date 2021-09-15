'''
mistake: 1.growth disperse
        2. do not update with death
        3. mutation_ID = [0] repeat with origin mutation_ID
  [Cancer growth model]
  [2020/2/28]

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

start_time = time.time()

### Basic parameter setting

Max_generation = 30
push_rate = 0  # the frequency of cancer cell push each others
# cancer growth space size
Max_ROW = 200
Max_COL = 200

background_gene = 1   # The background gene that cancer originally carried
mutation_rate = 0.0


die_divide_times = 500   # The max times of cell divides,  arrived this point the cell will die
Poisson_lambda = 10   # the mean value of Poisson distributions
birth_rate = 1
death_rate = 0.0
mutation_type2 = 1
mutation_type3 = 2
mutation_type4 = 3
mutation_type5 = 4

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
            print(count)
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
            poisson = Poisson()
            print('poisson = ',poisson)
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
                          max(all_mutation + agent[self.cp].mutation) + 1 + poisson))
                agent[self.cp].mutation = agent[self.cp].mutation + list(
                    range(max(all_mutation + daughter_cell.mutation) + 1,
                          max(all_mutation + daughter_cell.mutation) + 1 + poisson))
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
                          max(all_mutation + agent[self.cp].mutation) + 1 + poisson))
                agent[self.cp].mutation = agent[self.cp].mutation + list(
                    range(max(all_mutation + agent[newcell_pos].mutation) + 1,
                          max(all_mutation + agent[newcell_pos].mutation) + 1 + poisson))
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
        cut_point = []
        for i in range(Max_ROW):
            for j in range(Max_ROW):
                if cancer_matrix[i, j] != 0:
                    cut_point.append(i)
                    break
        cut_point1 = min(cut_point)
        cut_point2 = max(cut_point)

        gap = round((cut_point2 - cut_point1) / 5)
    # print('ccc', gap, cut_point, cut_point1, cut_point2)

    # cut the tissue into 5 piece

    cancer_num = np.sum(cancer_matrix != 0)
    cancer_num1 = np.sum(cancer_matrix[cut_point1:(cut_point1 + gap), :] != 0)
    cancer_num2 = np.sum(cancer_matrix[(cut_point1 + gap):(cut_point1 + 2 * gap), :] != 0)
    cancer_num3 = np.sum(cancer_matrix[(cut_point1 + 2 * gap):(cut_point1 + 3 * gap), :] != 0)
    cancer_num4 = np.sum(cancer_matrix[(cut_point1 + 3 * gap):(cut_point1 + 4 * gap), :] != 0)
    cancer_num5 = np.sum(cancer_matrix[(cut_point1 + 4 * gap):, :] != 0)


    ratio1 = mutation_num2/cancer_num
    ratio2 = mutation_num3 / cancer_num
    ratio3 = mutation_num4 / cancer_num
    ratio4 = mutation_num5 / cancer_num
    print('cancer number',cancer_num,'\n','mutation ratio2:',ratio1,'\n','mutation ratio3:',ratio2,'\n','mutation ratio4:',ratio3,'\n','mutation ratio5:',ratio4)

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
    vertical_x1 = []
    vertical_y1 = []
    vertical_x2 = []
    vertical_y2 = []
    vertical_x3 = []
    vertical_y3 = []
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
    era_x = []
    era_y = []
    for cut in range(0, len(cell_r)):
        x0 = cell_r[cut]
        y0 = cell_c[cut]

        if cut_point1 < x0 < cut_point1 + gap:
            horizon_x1.append(x0)
            horizon_y1.append(y0)

        if cut_point1 + gap <= x0 < cut_point1 + 2 * gap:
            horizon_x2.append(x0)
            horizon_y2.append(y0)

        if cut_point1 + 2 * gap <= x0 < cut_point1 + 3 * gap:
            horizon_x3.append(x0)
            horizon_y3.append(y0)
        if cut_point1 + 3 * gap <= x0 < cut_point1 + 4 * gap:
            horizon_x4.append(x0)
            horizon_y4.append(y0)
        if cut_point1 + 4 * gap <= x0 < cut_point1 + 5 * gap:
            horizon_x5.append(x0)
            horizon_y5.append(y0)

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
    VAF = list(map(lambda n: n / (cancer_num),
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
    for f in range(len(VAF_number)):
        VAF_cumulative.append(np.sum(VAF_number[0:f]))
    VAF_f = list(map(lambda c: 1 / c, VAF_mu))

    ### horizon slice of cancer tissue
    ### 1 slice
    for cut1 in range(0, len(horizon_x1)):
        x1 = horizon_x1[cut1]
        y1 = horizon_y1[cut1]
        mutation_ID1 = cell_dictionary[x1, y1].mutation  # get all the mutation ID in the dictionary

        all_mutation_ID1 += mutation_ID1  # Collect every mutation.ID in the all_mutation_ID

    result1 = Counter(all_mutation_ID1)  # Count frequency of each mutation ID
    count_mu1 = []  # empty list: record the mutation ID in the VAF
    count_times1 = []  # empty list: record the frequency of mutation ID in the VAF
    for i in result1:
        count_mu1.append(i)

    for j in count_mu1:
        count_times1.append(result1[j])

    # counter the mutation proberbility in the clip area
    VAF1 = list(map(lambda c: c / (2 * cancer_num1),
                    count_times1))  # Object function change to list, and calculate each element of count_times

    result_VAF1 = Counter(VAF1)  # Count each element in VAF, feedback the numbers

    VAF_mu1 = []  # empty list: record the mutation ID in the VAF
    VAF_times1 = []  # empty list: record the each ID frequency in the VAF
    for i1 in result_VAF1:
        VAF_mu1.append(i1)
    list.sort(VAF_mu1, reverse=1)
    for j1 in VAF_mu1:
        VAF_times1.append(result_VAF1[j1])

    VAF_cumulative1 = []
    VAF_number1 = list(VAF_times1)
    for f1 in range(len(VAF_number1)):
        VAF_cumulative1.append(np.sum(VAF_number1[0:f1]))
    VAF_f1 = list(map(lambda c: 1 / c, VAF_mu1))

    ### 2 slice
    for cut2 in range(0, len(horizon_x2)):
        x2 = horizon_x2[cut2]
        y2 = horizon_y2[cut2]
        mutation_ID2 = cell_dictionary[x2, y2].mutation  # get all the mutation ID in the dictionary

        all_mutation_ID2 += mutation_ID2  # Collect every mutation.ID in the all_mutation_ID

    result2 = Counter(all_mutation_ID2)  # Count frequency of each mutation ID
    count_mu2 = []  # empty list: record the mutation ID in the VAF
    count_times2 = []  # empty list: record the frequency of mutation ID in the VAF
    for i in result2:
        count_mu2.append(i)

    for j in count_mu2:
        count_times2.append(result2[j])

    # counter the mutation proberbility in the clip area
    VAF2 = list(map(lambda c: c / (2 * cancer_num2),
                    count_times2))  # Object function change to list, and calculate each element of count_times

    result_VAF2 = Counter(VAF2)  # Count each element in VAF, feedback the numbers

    VAF_mu2 = []  # empty list: record the mutation ID in the VAF
    VAF_times2 = []  # empty list: record the each ID frequency in the VAF
    for i2 in result_VAF2:
        VAF_mu2.append(i2)
    list.sort(VAF_mu2, reverse=1)
    for j2 in VAF_mu2:
        VAF_times2.append(result_VAF2[j2])

    VAF_cumulative2 = []
    VAF_number2 = list(VAF_times2)
    for f2 in range(len(VAF_number2)):
        VAF_cumulative2.append(np.sum(VAF_number2[0:f2]))
    VAF_f2 = list(map(lambda c: 1 / c, VAF_mu2))

    ## 3 slice
    for cut3 in range(0, len(horizon_x3)):
        x3 = horizon_x3[cut3]
        y3 = horizon_y3[cut3]
        mutation_ID3 = cell_dictionary[x3, y3].mutation  # get all the mutation ID in the dictionary

        all_mutation_ID3 += mutation_ID3  # Collect every mutation.ID in the all_mutation_ID

    result3 = Counter(all_mutation_ID3)  # Count frequency of each mutation ID
    count_mu3 = []  # empty list: record the mutation ID in the VAF
    count_times3 = []  # empty list: record the frequency of mutation ID in the VAF
    for i in result3:
        count_mu3.append(i)

    for j in count_mu3:
        count_times3.append(result3[j])

    # counter the mutation proberbility in the clip area
    VAF3 = list(map(lambda c: c / (2 * cancer_num3),
                    count_times3))  # Object function change to list, and calculate each element of count_times

    result_VAF3 = Counter(VAF3)  # Count each element in VAF, feedback the numbers

    VAF_mu3 = []  # empty list: record the mutation ID in the VAF
    VAF_times3 = []  # empty list: record the each ID frequency in the VAF
    for i3 in result_VAF3:
        VAF_mu3.append(i3)
    list.sort(VAF_mu3, reverse=1)
    for j3 in VAF_mu3:
        VAF_times3.append(result_VAF3[j3])

    VAF_cumulative3 = []
    VAF_number3 = list(VAF_times3)
    for f3 in range(len(VAF_number3)):
        VAF_cumulative3.append(np.sum(VAF_number3[0:f3]))
    VAF_f3 = list(map(lambda c: 1 / c, VAF_mu3))

    # 4 slice
    for cut4 in range(0, len(horizon_x4)):
        x4 = horizon_x4[cut4]
        y4 = horizon_y4[cut4]
        mutation_ID4 = cell_dictionary[x4, y4].mutation  # get all the mutation ID in the dictionary

        all_mutation_ID4 += mutation_ID4  # Collect every mutation.ID in the all_mutation_ID

    result4 = Counter(all_mutation_ID4)  # Count frequency of each mutation ID
    count_mu4 = []  # empty list: record the mutation ID in the VAF
    count_times4 = []  # empty list: record the frequency of mutation ID in the VAF
    for i in result4:
        count_mu4.append(i)

    for j in count_mu4:
        count_times4.append(result4[j])

    # counter the mutation proberbility in the clip area
    VAF4 = list(map(lambda c: c / (2 * cancer_num4),
                    count_times4))  # Object function change to list, and calculate each element of count_times

    result_VAF4 = Counter(VAF4)  # Count each element in VAF, feedback the numbers

    VAF_mu4 = []  # empty list: record the mutation ID in the VAF
    VAF_times4 = []  # empty list: record the each ID frequency in the VAF
    for i4 in result_VAF4:
        VAF_mu4.append(i4)
    list.sort(VAF_mu4, reverse=1)
    for j4 in VAF_mu4:
        VAF_times4.append(result_VAF4[j4])

    VAF_cumulative4 = []
    VAF_number4 = list(VAF_times4)
    for f4 in range(len(VAF_number4)):
        VAF_cumulative4.append(np.sum(VAF_number4[0:f4]))
    VAF_f4 = list(map(lambda c: 1 / c, VAF_mu4))

    # slice 5
    for cut5 in range(0, len(horizon_x5)):
        x5 = horizon_x5[cut5]
        y5 = horizon_y5[cut5]
        mutation_ID5 = cell_dictionary[x5, y5].mutation  # get all the mutation ID in the dictionary

        all_mutation_ID5 += mutation_ID5  # Collect every mutation.ID in the all_mutation_ID

    result5 = Counter(all_mutation_ID5)  # Count frequency of each mutation ID
    count_mu5 = []  # empty list: record the mutation ID in the VAF
    count_times5 = []  # empty list: record the frequency of mutation ID in the VAF
    for i in result5:
        count_mu5.append(i)

    for j in count_mu5:
        count_times5.append(result5[j])

    # counter the mutation proberbility in the clip area
    VAF5 = list(map(lambda c: c / (2 * cancer_num5),
                    count_times5))  # Object function change to list, and calculate each element of count_times

    result_VAF5 = Counter(VAF5)  # Count each element in VAF, feedback the numbers

    VAF_mu5 = []  # empty list: record the mutation ID in the VAF
    VAF_times5 = []  # empty list: record the each ID frequency in the VAF
    for i5 in result_VAF5:
        VAF_mu5.append(i5)
    list.sort(VAF_mu5, reverse=1)
    for j5 in VAF_mu5:
        VAF_times5.append(result_VAF5[j5])

    VAF_cumulative5 = []
    VAF_number5 = list(VAF_times5)
    for f5 in range(len(VAF_number5)):
        VAF_cumulative5.append(np.sum(VAF_number5[0:f5]))
    VAF_f5 = list(map(lambda c: 1 / c, VAF_mu5))

    end_time = time.time()
    run_time = end_time - start_time

    end_time = time.time()
    run_time = end_time - start_time

    print('Growth Run time is: {}'.format(run_time), '\n', 'Max Mutation', '\n', 'mutation ID number per cell',
          len(all_mutation_ID) / cancer_num, '\n', 'generation times:{}'.format(Max_generation), '\n',
          'Cancer Number is: {}'.format(cancer_num), '\n', 'Mutation 2 nember : ', mutation_num2, '\n',
          'Mutation 3 nember : ', mutation_num3, '\n', 'Mutation 4 nember : ', mutation_num4, '\n',
          'Mutation 5 nember : ', mutation_num5)
    print('number', cancer_num1, cancer_num2, cancer_num3, cancer_num4, cancer_num5, '\n', 'In total', cancer_num,
          cancer_num1 + cancer_num2 + cancer_num3 + cancer_num4 + cancer_num5)
    plt.figure('VAF Cumulative')
    plt.subplot(151)
    plt.scatter(VAF_f1, VAF_cumulative1)
    plt.xlabel('Inverse allelic frequency 1/f')
    plt.ylabel("Cumulative Number of Mutations M(f)")
    plt.subplot(152)
    plt.scatter(VAF_f2, VAF_cumulative2)
    plt.xlabel('Inverse allelic frequency 1/f')
    plt.subplot(153)
    plt.scatter(VAF_f3, VAF_cumulative3)
    plt.xlabel('Inverse allelic frequency 1/f')
    plt.subplot(154)
    plt.scatter(VAF_f4, VAF_cumulative4)
    plt.xlabel('Inverse allelic frequency 1/f')
    plt.subplot(155)
    plt.scatter(VAF_f5, VAF_cumulative5)
    plt.xlabel('Inverse allelic frequency 1/f')

    plt.figure('Horizon VAF')

    plt.bar(VAF_mu, VAF_times, width=0.005, color='seagreen')
    plt.xlabel("VAF")
    plt.ylabel("Number of Mutation")

    plt.subplot(151)
    plt.bar(VAF_mu1, VAF_times1, width=0.005, color='seagreen')
    plt.xlabel("VAF(A)")
    plt.ylabel("Number of Mutation")

    plt.subplot(152)
    plt.bar(VAF_mu2, VAF_times2, width=0.005, color='seagreen')
    plt.xlabel("VAF(B)")

    plt.subplot(153)
    plt.bar(VAF_mu3, VAF_times3, width=0.005, color='seagreen')
    plt.xlabel('VAF(C)')

    plt.subplot(154)
    plt.bar(VAF_mu4, VAF_times4, width=0.005, color='seagreen')
    plt.xlabel('VAF(D)')

    plt.subplot(155)
    plt.bar(VAF_mu5, VAF_times5, width=0.01, color='seagreen')
    plt.xlabel('VAF(E)')

    plt.figure('Whole plot')
    plt.bar(VAF_mu, VAF_times, width=0.01, color='seagreen')
    plt.xlabel('VAF')
    plt.ylim(0,200)
    plt.ylabel("Number of Mutation")
    # plt.yscale('log')



    plt.figure('cumulative')

    plt.scatter(VAF_f, VAF_cumulative)
    plt.xlabel('Inverse allelic frequency 1/f')
    plt.ylabel("Cumulative Number of Mutations M(f)")

    map = matplotlib.colors.ListedColormap(
        ['#FFFFFF', '#BA4EF9', '#BA4EF9', '#E8E20A', '#EF6273',
         '#EFABAB',] )
    plt.figure('cancer growth')
    plt.title('Cancer Number is: {}'.format(cancer_num))
    plt.imshow(cancer_matrix, cmap=map)
    plt.figure('Cell growth')
    plt.title('Cancer Number is: {}'.format(cancer_num))
    plt.imshow(cell_matrix, cmap=map)
    plt.show()
    plt.ion()




