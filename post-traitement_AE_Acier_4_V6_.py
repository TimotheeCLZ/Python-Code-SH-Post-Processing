#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# T. Cullaz

import math
import matplotlib.pyplot as plt
import matplotlib as mpl
#from cycler import cycler
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['lines.linestyle'] = '-'
#mpl.rcParams['axes.prop_cycle'] = cycler(color=['b', 'g', 'r', 'y'])
mpl.rcParams['text.usetex'] = False
mpl.rcParams['axes.labelsize'] = 15
mpl.rcParams['axes.titlesize'] = 15
mpl.rcParams['axes.labelpad'] = 5
mpl.rcParams['figure.figsize'] = 8, 6
props = dict(facecolor='gray', alpha=0.25)

#mpl.rcParams[''] = 
#mpl.rcParams[''] = 
#mpl.rcParams[''] = 
#mpl.rcParams[''] = 
#mpl.rcParams[''] = 
#mpl.rcParams[''] = 
#mpl.rcParams[''] = 
#mpl.rcParams[''] = 
#mpl.rcParams[''] = 

import numpy as np
import os
import sys
import pandas as pd
from pandas.io.parsers import TextParser
import statistics
from scipy import signal, optimize
import matplotlib.cm as cm
import csv


#---------------------------------------------------- Fichier ----------------------------------------------------#

# Chargement des fichers
filename = "Steel "
filename_1 = "Steel"
# filename_2 = "NiTi 04_02"


contrainte_1     = list(range(80, 305, 15))      # contraintes 
contraintes_2 = list(range(300, 520, 20))
contrainte = contrainte_1 + contraintes_2

freq            = 5.5                           # Hz, fréquence d'acquisition des images
time            = 112.007                         # s, durée d'un palier

conditions = 'T=10°C - R=10 - f=30Hz' 

def conv(text):
  return text.decode("utf-8").replace(",", ".")

(t1, T1) = np.loadtxt('AE_acier_rotule_1500cycles_80-500MPa_4_corrige.csv', unpack = True, delimiter=';', skiprows=1)
# (t2, T2) = np.loadtxt(filename_2+'_corrige.csv', unpack = True, delimiter=';', skiprows=1)
(t1_raw, T1_epp_raw, T1_M1, T1_M2) = np.loadtxt('AE_acier_rotule_1500cycles_80-500MPa_4.csv', unpack = True, delimiter='\t', skiprows=1, converters={0: conv, 1: conv, 2: conv, 3: conv})
# (t2_raw, T2_epp_raw, T2_M1, T2_M2) = np.loadtxt(filename_2+'.csv', unpack = True, delimiter='\t', skiprows=1, converters={0: conv, 1: conv, 2: conv, 3: conv})

#---------------------------------------------------- GRAPH Comparison corriged/raw curve ----------------------------------------------------#
# plt.figure (1)
# plt.grid()
# plt.plot(t1,T1, label = 'corriged')
# plt.plot(t1, T1_epp_raw, label='raw')
# plt.legend()

T_M_moy = (T1_M1+T1_M2)/2
AE = T1_epp_raw - T_M_moy

# plt.figure (2)
# plt.grid()
# plt.plot(t1,[A- T1[0] for A in T1], label = 'AE_corriged')
# plt.plot(t1, [A - AE[0] for A in AE], label='AE_raw')
# plt.legend()
# plt.show()

#---------------------------------------------------- Adjustment curve ----------------------------------------------------#
t1 = (t1/1000)-15.58
# t2 = (t2/1000)-30.35
# T2 = [A + 1.281 for A in T2]

# t1_adj = []
# T1_adj =[]
# t2_adj = []
# T2_adj =[]

# for n, m in zip(t1, T1):
#     if n <= 3191.2:
#         t1_adj.append(n)
#         T1_adj.append(m)

# for n, m in zip(t2,T2):
#     if 0 <= n < 2156:
#         t2_adj.append(n)
#         T2_adj.append(m)

# t2_adj = [A + t1_adj[-1] for A in t2_adj]

t = t1
T = [A- T1[0] for A in T1]
AE = [A - AE[0] for A in AE]
#---------------------------------------------------- GRAPH Adjustment curve ----------------------------------------------------#
# fig3 = plt.figure (3)
# ax = fig3.add_subplot(111)
# plt.title(filename)
# plt.grid()
# plt.plot(t, T, 'navy',label = filename +r" - Sample Temperature - PP1")
# plt.plot(t, AE, 'b',label = filename +r" - Sample Temperature - PP2")
# plt.ylabel(r"Temperature elevation,  $\theta$ (°C)")
# plt.xlabel(r"Time, $t$ (s)")
# plt.legend()
# textstr = '\n'.join((r'$f= 30Hz$',r'$R = 10 $',r'$T= 10°C$'))
# ax.text(0.05, 0.90, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
# # plt.show()

#---------------------------------------------------- Dictionnary ----------------------------------------------------#
p_n             = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
p_txt           = {'p1':[],'p2':[],'p3':[],'p4':[],'p5':[],'p6':[],'p7':[],'p8':[],'p9':[],'p10':[],'p11':[],'p12':[],'p13':[],'p14':[],'p15':[],'p16':[],'p17':[],'p18':[],'p19':[],'p20':[],'p21':[],'p22':[],'p23':[],'p24':[],'p25':[],'p26':[]}
p_x             = {'p1':[],'p2':[],'p3':[],'p4':[],'p5':[],'p6':[],'p7':[],'p8':[],'p9':[],'p10':[],'p11':[],'p12':[],'p13':[],'p14':[],'p15':[],'p16':[],'p17':[],'p18':[],'p19':[],'p20':[],'p21':[],'p22':[],'p23':[],'p24':[],'p25':[],'p26':[]}
p_y_pp1         = {'p1':[],'p2':[],'p3':[],'p4':[],'p5':[],'p6':[],'p7':[],'p8':[],'p9':[],'p10':[],'p11':[],'p12':[],'p13':[],'p14':[],'p15':[],'p16':[],'p17':[],'p18':[],'p19':[],'p20':[],'p21':[],'p22':[],'p23':[],'p24':[],'p25':[],'p26':[]}
p_y_pp2         = {'p1':[],'p2':[],'p3':[],'p4':[],'p5':[],'p6':[],'p7':[],'p8':[],'p9':[],'p10':[],'p11':[],'p12':[],'p13':[],'p14':[],'p15':[],'p16':[],'p17':[],'p18':[],'p19':[],'p20':[],'p21':[],'p22':[],'p23':[],'p24':[],'p25':[],'p26':[]}
p_time          = [A * time for A in p_n]

Theta_bar_end   = {'p1':[], 'p2':[], 'p3':[], 'p4':[], 'p5':[], 'p6':[], 'p7':[], 'p8':[], 'p9':[], 'p10':[], 'p11':[], 'p12':[], 'p13':[], 'p14':[], 'p15':[],'p16':[],'p17':[],'p18':[],'p19':[],'p20':[],'p21':[],'p22':[],'p23':[],'p24':[],'p25':[],'p26':[]}


gra_x           = {'p1':[85,100], 'p2':[85,100], 'p3':[85,100], 'p4':[85,100], 'p5':[85,100], 'p6':[85,100], 'p7':[85,100], 'p8':[85,100], 'p9':[85,100], 'p10':[85,100], 'p11':[85,100], 'p12':[85,100], 'p13':[85,100], 'p14':[85,100], 'p15':[85,100],'p16':[85,100],'p17':[85,100],'p18':[85,100],'p19':[85,100],'p20':[85,100],'p21':[85,100],'p22':[85,100],'p23':[85,100],'p24':[85,100],'p25':[85,100],'p26':[85,100]}
gra_y           = {'p1':[], 'p2':[], 'p3':[], 'p4':[], 'p5':[], 'p6':[], 'p7':[], 'p8':[], 'p9':[], 'p10':[], 'p11':[], 'p12':[], 'p13':[], 'p14':[], 'p15':[],'p16':[],'p17':[],'p18':[],'p19':[],'p20':[],'p21':[],'p22':[],'p23':[],'p24':[],'p25':[],'p26':[]}

dT_exp1         = {'p1':[], 'p2':[], 'p3':[], 'p4':[], 'p5':[], 'p6':[], 'p7':[], 'p8':[], 'p9':[], 'p10':[], 'p11':[], 'p12':[], 'p13':[], 'p14':[], 'p15':[],'p16':[],'p17':[],'p18':[],'p19':[],'p20':[],'p21':[],'p22':[],'p23':[],'p24':[],'p25':[],'p26':[]}
couplage_ampl   = {'p1':[], 'p2':[], 'p3':[], 'p4':[], 'p5':[], 'p6':[], 'p7':[], 'p8':[], 'p9':[], 'p10':[], 'p11':[], 'p12':[], 'p13':[], 'p14':[], 'p15':[],'p16':[],'p17':[],'p18':[],'p19':[],'p20':[],'p21':[],'p22':[],'p23':[],'p24':[],'p25':[],'p26':[]}

#---------------------------------------------------- Levels building ----------------------------------------------------#
print("\n1. Levels determination")
for n in range(len(t)):                                
    if t[n] <= p_time[0]:
        p_x['p1'].append(t[n])
        p_y_pp1['p1'].append(T[n])      
        p_y_pp2['p1'].append(AE[n])                                            
    if p_time[0] < t[n] <= p_time[1]:
        p_x['p2'].append(t[n])
        p_y_pp1['p2'].append(T[n])      
        p_y_pp2['p2'].append(AE[n])                        
    if p_time[1] < t[n] <= p_time[2]:
        p_x['p3'].append(t[n])
        p_y_pp1['p3'].append(T[n])      
        p_y_pp2['p3'].append(AE[n]) 
    if p_time[2] < t[n] <= p_time[3]:
        p_x['p4'].append(t[n])
        p_y_pp1['p4'].append(T[n])      
        p_y_pp2['p4'].append(AE[n]) 
    if p_time[3] < t[n] <= p_time[4]:
        p_x['p5'].append(t[n])
        p_y_pp1['p5'].append(T[n])      
        p_y_pp2['p5'].append(AE[n]) 
    if p_time[4] < t[n] <= p_time[5]:
        p_x['p6'].append(t[n])
        p_y_pp1['p6'].append(T[n])      
        p_y_pp2['p6'].append(AE[n]) 
    if p_time[5] < t[n] <= p_time[6]:
        p_x['p7'].append(t[n])
        p_y_pp1['p7'].append(T[n])      
        p_y_pp2['p7'].append(AE[n]) 
    if p_time[6] < t[n] <= p_time[7]:
        p_x['p8'].append(t[n])
        p_y_pp1['p8'].append(T[n])      
        p_y_pp2['p8'].append(AE[n]) 
    if p_time[7] < t[n] <= p_time[8]:
        p_x['p9'].append(t[n])
        p_y_pp1['p9'].append(T[n])      
        p_y_pp2['p9'].append(AE[n]) 
    if p_time[8] < t[n] <= p_time[9]:
        p_x['p10'].append(t[n])
        p_y_pp1['p10'].append(T[n])      
        p_y_pp2['p10'].append(AE[n]) 
    if p_time[9] < t[n] <= p_time[10]:
        p_x['p11'].append(t[n])
        p_y_pp1['p11'].append(T[n])      
        p_y_pp2['p11'].append(AE[n]) 
    if p_time[10] < t[n] <= p_time[11]:
        p_x['p12'].append(t[n])
        p_y_pp1['p12'].append(T[n])      
        p_y_pp2['p12'].append(AE[n]) 
    if p_time[11] < t[n] <= p_time[12]:
        p_x['p13'].append(t[n])
        p_y_pp1['p13'].append(T[n])      
        p_y_pp2['p13'].append(AE[n]) 
    if p_time[12] < t[n] <= p_time[13]:
        p_x['p14'].append(t[n])
        p_y_pp1['p14'].append(T[n])      
        p_y_pp2['p14'].append(AE[n]) 
    if p_time[13] < t[n] <= p_time[14]:
        p_x['p15'].append(t[n])
        p_y_pp1['p15'].append(T[n])      
        p_y_pp2['p15'].append(AE[n]) 
    if p_time[14] < t[n] <= p_time[15]:
        p_x['p16'].append(t[n])
        p_y_pp1['p16'].append(T[n])      
        p_y_pp2['p16'].append(AE[n]) 
    if p_time[15] < t[n] <= p_time[16]:
        p_x['p17'].append(t[n])
        p_y_pp1['p17'].append(T[n])      
        p_y_pp2['p17'].append(AE[n]) 
    if p_time[16] < t[n] <= p_time[17]:
        p_x['p18'].append(t[n])
        p_y_pp1['p18'].append(T[n])      
        p_y_pp2['p18'].append(AE[n]) 
    if p_time[17] < t[n] <= p_time[18]:
        p_x['p19'].append(t[n])
        p_y_pp1['p19'].append(T[n])      
        p_y_pp2['p19'].append(AE[n]) 
    if p_time[18] < t[n] <= p_time[19]:
        p_x['p20'].append(t[n])
        p_y_pp1['p20'].append(T[n])      
        p_y_pp2['p20'].append(AE[n]) 
    if p_time[19] < t[n] <= p_time[20]:
        p_x['p21'].append(t[n])
        p_y_pp1['p21'].append(T[n])      
        p_y_pp2['p21'].append(AE[n]) 
    if p_time[20] < t[n] <= p_time[21]:
        p_x['p22'].append(t[n])
        p_y_pp1['p22'].append(T[n])      
        p_y_pp2['p22'].append(AE[n]) 
    if p_time[21] < t[n] <= p_time[22]:
        p_x['p23'].append(t[n])
        p_y_pp1['p23'].append(T[n])      
        p_y_pp2['p23'].append(AE[n]) 
    if p_time[22] < t[n] <= p_time[23]:
        p_x['p24'].append(t[n])
        p_y_pp1['p24'].append(T[n])      
        p_y_pp2['p24'].append(AE[n]) 
    if p_time[23] < t[n] <= p_time[24]:
        p_x['p25'].append(t[n])
        p_y_pp1['p25'].append(T[n])      
        p_y_pp2['p25'].append(AE[n]) 
    if p_time[24] < t[n] <= p_time[25]:
        p_x['p26'].append(t[n])
        p_y_pp1['p26'].append(T[n])      
        p_y_pp2['p26'].append(AE[n]) 

for n, m in  zip(p_txt.keys(), range(len(p_n))):
    if n == 'p1':
        p_x[n] = [A - 0 for A in p_x[n]]
        p_y_pp1[n] = [A - p_y_pp1[n][0] - 0.0213 for A in p_y_pp1[n]]
        p_y_pp2[n] = [A - p_y_pp2[n][0] - 0.0213 for A in p_y_pp2[n]]
    else:
        p_x[n] = [A - p_time[m-1] for A in p_x[n]]
        p_y_pp1[n] = [A - p_y_pp1[n][0] for A in p_y_pp1[n]]
        p_y_pp2[n] = [A - p_y_pp2[n][0] for A in p_y_pp2[n]]


#---------------------------------------------------- GRAPH Paliers Simples  ----------------------------------------------------#
for n in p_n:                                
    fig = plt.figure('Level - ' + str(n))
    ax = fig.add_subplot(111)
    plt.title(filename + ' - Level '+ str(n) + ' - Sigma_max = ' + str(contrainte[n-1]) +'MPa')
    plt.grid()
    plt.plot(p_x['p'+str(n)], p_y_pp1['p'+str(n)], 'navy',label = filename +r" - PP1 - dérive lineraire des mors")
    plt.plot(p_x['p'+str(n)], p_y_pp2['p'+str(n)], 'b',label = filename +r" - PP2 - température des mors")
    plt.xlabel(r'Time, $t$ (s)')
    plt.ylabel('Temperature elevation, 'r'$\theta$' '(°C)')
    plt.legend()
    textstr = '\n'.join((r'$f= 30Hz$',r'$R = 10 $',r'$T= 10°C$'))
    ax.text(0.85, 0.90, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
    #plt.savefig(filename + '/' + filename + ' - Level '+ str(n) + ' - Sigma_max = ' + str(contrainte[n-1]) +'MPa.pdf')
    #plt.savefig(filename + '/' + filename + ' - Level '+ str(n) + ' - Sigma_max = ' + str(contrainte[n-1]) +'MPa.png')
plt.show()

# Decalage pour l'image filtre
# p_y['p16'] = [A +0.184 for A in p_y['p16']]
# p_y['p17'] = [A + 0.036 for A in p_y['p17']]
# p_y['p19'] = [A -0.006 for A in p_y['p19']]

#----------------------------------------------------  Identification ----------------------------------------------------#
print('\n2. Identification')
print('2.1    Stabilized theta bar :')
p_x_heat        = {'p1':[], 'p2':[], 'p3':[], 'p4':[], 'p5':[], 'p6':[], 'p7':[], 'p8':[], 'p9':[], 'p10':[], 'p11':[], 'p12':[], 'p13':[], 'p14':[], 'p15':[],'p16':[],'p17':[],'p18':[],'p19':[],'p20':[],'p21':[],'p22':[],'p23':[],'p24':[],'p25':[],'p26':[]}
p_y_heat_pp1    = {'p1':[], 'p2':[], 'p3':[], 'p4':[], 'p5':[], 'p6':[], 'p7':[], 'p8':[], 'p9':[], 'p10':[], 'p11':[], 'p12':[], 'p13':[], 'p14':[], 'p15':[],'p16':[],'p17':[],'p18':[],'p19':[],'p20':[],'p21':[],'p22':[],'p23':[],'p24':[],'p25':[],'p26':[]}
p_y_heat_pp2    = {'p1':[], 'p2':[], 'p3':[], 'p4':[], 'p5':[], 'p6':[], 'p7':[], 'p8':[], 'p9':[], 'p10':[], 'p11':[], 'p12':[], 'p13':[], 'p14':[], 'p15':[],'p16':[],'p17':[],'p18':[],'p19':[],'p20':[],'p21':[],'p22':[],'p23':[],'p24':[],'p25':[],'p26':[]}
p_x_cool        = {'p1':[], 'p2':[], 'p3':[], 'p4':[], 'p5':[], 'p6':[], 'p7':[], 'p8':[], 'p9':[], 'p10':[], 'p11':[], 'p12':[], 'p13':[], 'p14':[], 'p15':[],'p16':[],'p17':[],'p18':[],'p19':[],'p20':[],'p21':[],'p22':[],'p23':[],'p24':[],'p25':[],'p26':[]}
p_y_cool_pp1        = {'p1':[], 'p2':[], 'p3':[], 'p4':[], 'p5':[], 'p6':[], 'p7':[], 'p8':[], 'p9':[], 'p10':[], 'p11':[], 'p12':[], 'p13':[], 'p14':[], 'p15':[],'p16':[],'p17':[],'p18':[],'p19':[],'p20':[],'p21':[],'p22':[],'p23':[],'p24':[],'p25':[],'p26':[]}
p_y_cool_pp2        = {'p1':[], 'p2':[], 'p3':[], 'p4':[], 'p5':[], 'p6':[], 'p7':[], 'p8':[], 'p9':[], 'p10':[], 'p11':[], 'p12':[], 'p13':[], 'p14':[], 'p15':[],'p16':[],'p17':[],'p18':[],'p19':[],'p20':[],'p21':[],'p22':[],'p23':[],'p24':[],'p25':[],'p26':[]}
p_x_end         = {'p1':[], 'p2':[], 'p3':[], 'p4':[], 'p5':[], 'p6':[], 'p7':[], 'p8':[], 'p9':[], 'p10':[], 'p11':[], 'p12':[], 'p13':[], 'p14':[], 'p15':[],'p16':[],'p17':[],'p18':[],'p19':[],'p20':[],'p21':[],'p22':[],'p23':[],'p24':[],'p25':[],'p26':[]}
p_y_end_pp1         = {'p1':[], 'p2':[], 'p3':[], 'p4':[], 'p5':[], 'p6':[], 'p7':[], 'p8':[], 'p9':[], 'p10':[], 'p11':[], 'p12':[], 'p13':[], 'p14':[], 'p15':[],'p16':[],'p17':[],'p18':[],'p19':[],'p20':[],'p21':[],'p22':[],'p23':[],'p24':[],'p25':[],'p26':[]}
p_y_end_pp2         = {'p1':[], 'p2':[], 'p3':[], 'p4':[], 'p5':[], 'p6':[], 'p7':[], 'p8':[], 'p9':[], 'p10':[], 'p11':[], 'p12':[], 'p13':[], 'p14':[], 'p15':[],'p16':[],'p17':[],'p18':[],'p19':[],'p20':[],'p21':[],'p22':[],'p23':[],'p24':[],'p25':[],'p26':[]}

Temp_stab = []

for i,n in zip(p_txt.keys(), range(len(p_txt))):
    for n, m, l in zip(p_x[i], p_y_pp1[i], p_y_pp2[i]):        
        if 0<= n < 45:                                   #Start heating          
            p_x_heat[i].append(n)
            p_y_heat_pp1[i].append(m)
            p_y_heat_pp2[i].append(l)
        if 40<= n < 48:                                #Stabilized theta bar
            p_x_end[i].append(n)
            p_y_end_pp1[i].append(m)
            p_y_end_pp2[i].append(l)
        if 50 <= n :                          #Start cooling            
            p_x_cool[i].append(n)
            p_y_cool_pp1[i].append(m)
            p_y_cool_pp2[i].append(l)

    # X_pp1 = statistics.mean(p_y_end_pp1[i])
    # X_pp2 = statistics.mean(p_y_end_pp2[i])
    #X = float('%1g' % X)
    # Theta_bar_end[i].append(X)
    # gra_y[i].append(X)
    # print(i, Theta_bar_end[i])

#---------------------------------------------------- GRAPH Paliers + c + theta bar + r  ----------------------------------------------------#
# for n in p_n:                                
#     fig = plt.figure(n)
#     ax = fig.add_subplot(111)
#     plt.title(filename + r" - Level "+ str(n) + r" - $\sigma_{max}$ = " + str(contrainte[n-1]) +'MPa')
#     plt.grid()
#     # plt.plot(p_x['p'+str(n)], p_y['p'+str(n)], 'sk', label = 'Temperature elevation, 'r'$\theta$' )
#     # plt.plot(p_x['p'+str(n)], p_y['p'+str(n)], 'navy', label = 'Temperature elevation, 'r'$\theta$' )
#     plt.plot(p_x_heat['p'+str(n)], p_y_heat_pp1['p'+str(n)], 'firebrick')
#     plt.plot(p_x_cool['p'+str(n)], p_y_cool_pp1['p'+str(n)], 'b')
#     plt.plot(p_x_end['p'+str(n)], p_y_end_pp1['p'+str(n)], 'k')
#     plt.plot(p_x_heat['p'+str(n)], p_y_heat_pp2['p'+str(n)], 'r--')
#     plt.plot(p_x_cool['p'+str(n)], p_y_cool_pp2['p'+str(n)], 'b--')
#     plt.plot(p_x_end['p'+str(n)], p_y_end_pp2['p'+str(n)], 'k--')
#     # plt.plot([85, 100],[Theta_bar_end['p'+str(n)], Theta_bar_end['p'+str(n)]], 'r', label ='Temperature elevation, 'r"$\bar{\theta}^{0D}$")
#     plt.xlabel('Time, 'r"$t$" ' (s)')
#     plt.ylabel('Temperature elevation, 'r'$\theta$' ' (°C)')
#     plt.legend()
#     textstr = '\n'.join((r'$f= 30Hz$',r'$R = 10 $',r'$T= 10°C$'))
#     ax.text(0.85, 0.85, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
#     #plt.savefig(filename + '/' + filename + ' - Level '+ str(n) + ' - Sigma_max_ ' + str(contrainte[n-1]) +'MPa.pdf')
#     #plt.savefig(filename + '/' + filename + ' - Level '+ str(n) + ' - Sigma_max_ ' + str(contrainte[n-1]) +'MPa.png')
# plt.show()

#----------------------------------------------------  Identification - EXPO 1 ----------------------------------------------------#
print('\n2.2    Exponential Curve n°1')
x0              = {'p1':[0],'p2':[0],'p3':[0],'p4':[0.22],'p5':[0.50],'p6':[0.5],'p7':[0.50],'p8':[0.50],'p9':[0.5],'p10':[0.5],'p11':[0.65],'p12':[0.65],'p13':[1.016],'p14':[0.75],'p15':[1.00],'p16':[0.87],'p17':[0.85],'p18':[1.036],'p19':[1.15],'p20':[1.57],'p21':[1.37],'p22':[2.67],'p23':[2.8],'p24':[2.7],'p25':[2.48],'p26':[2.86]}
p_x_exp1        = {'p1':[], 'p2':[], 'p3':[], 'p4':[], 'p5':[], 'p6':[], 'p7':[], 'p8':[], 'p9':[], 'p10':[], 'p11':[], 'p12':[], 'p13':[], 'p14':[], 'p15':[],'p16':[],'p17':[],'p18':[],'p19':[],'p20':[],'p21':[],'p22':[],'p23':[],'p24':[],'p25':[],'p26':[]}
p_y_exp1        = {'p1':[], 'p2':[], 'p3':[], 'p4':[], 'p5':[], 'p6':[], 'p7':[], 'p8':[], 'p9':[], 'p10':[], 'p11':[], 'p12':[], 'p13':[], 'p14':[], 'p15':[],'p16':[],'p17':[],'p18':[],'p19':[],'p20':[],'p21':[],'p22':[],'p23':[],'p24':[],'p25':[],'p26':[]}
p_x_exp1_20s    = {'p1':[], 'p2':[], 'p3':[], 'p4':[], 'p5':[], 'p6':[], 'p7':[], 'p8':[], 'p9':[], 'p10':[], 'p11':[], 'p12':[], 'p13':[], 'p14':[], 'p15':[],'p16':[],'p17':[],'p18':[],'p19':[],'p20':[],'p21':[],'p22':[],'p23':[],'p24':[],'p25':[],'p26':[]}
p_y_exp1_20s    = {'p1':[], 'p2':[], 'p3':[], 'p4':[], 'p5':[], 'p6':[], 'p7':[], 'p8':[], 'p9':[], 'p10':[], 'p11':[], 'p12':[], 'p13':[], 'p14':[], 'p15':[],'p16':[],'p17':[],'p18':[],'p19':[],'p20':[],'p21':[],'p22':[],'p23':[],'p24':[],'p25':[],'p26':[]}
coef_a1          = {'p1':[], 'p2':[], 'p3':[], 'p4':[], 'p5':[], 'p6':[], 'p7':[], 'p8':[], 'p9':[], 'p10':[], 'p11':[], 'p12':[], 'p13':[], 'p14':[], 'p15':[],'p16':[],'p17':[],'p18':[],'p19':[],'p20':[],'p21':[],'p22':[],'p23':[],'p24':[],'p25':[],'p26':[]}
coef_b1          = {'p1':[], 'p2':[], 'p3':[], 'p4':[], 'p5':[], 'p6':[], 'p7':[], 'p8':[], 'p9':[], 'p10':[], 'p11':[], 'p12':[], 'p13':[], 'p14':[], 'p15':[],'p16':[],'p17':[],'p18':[],'p19':[],'p20':[],'p21':[],'p22':[],'p23':[],'p24':[],'p25':[],'p26':[]}
coef_c1          = {'p1':[], 'p2':[], 'p3':[], 'p4':[], 'p5':[], 'p6':[], 'p7':[], 'p8':[], 'p9':[], 'p10':[], 'p11':[], 'p12':[], 'p13':[], 'p14':[], 'p15':[],'p16':[],'p17':[],'p18':[],'p19':[],'p20':[],'p21':[],'p22':[],'p23':[],'p24':[],'p25':[],'p26':[]}
coef_a2          = {'p1':[], 'p2':[], 'p3':[], 'p4':[], 'p5':[], 'p6':[], 'p7':[], 'p8':[], 'p9':[], 'p10':[], 'p11':[], 'p12':[], 'p13':[], 'p14':[], 'p15':[],'p16':[],'p17':[],'p18':[],'p19':[],'p20':[],'p21':[],'p22':[],'p23':[],'p24':[],'p25':[],'p26':[]}
coef_b2          = {'p1':[], 'p2':[], 'p3':[], 'p4':[], 'p5':[], 'p6':[], 'p7':[], 'p8':[], 'p9':[], 'p10':[], 'p11':[], 'p12':[], 'p13':[], 'p14':[], 'p15':[],'p16':[],'p17':[],'p18':[],'p19':[],'p20':[],'p21':[],'p22':[],'p23':[],'p24':[],'p25':[],'p26':[]}
coef_c2          = {'p1':[], 'p2':[], 'p3':[], 'p4':[], 'p5':[], 'p6':[], 'p7':[], 'p8':[], 'p9':[], 'p10':[], 'p11':[], 'p12':[], 'p13':[], 'p14':[], 'p15':[],'p16':[],'p17':[],'p18':[],'p19':[],'p20':[],'p21':[],'p22':[],'p23':[],'p24':[],'p25':[],'p26':[]}
expo1_pp1_Tbar  = {'p1':[], 'p2':[], 'p3':[], 'p4':[], 'p5':[], 'p6':[], 'p7':[], 'p8':[], 'p9':[], 'p10':[], 'p11':[], 'p12':[], 'p13':[], 'p14':[], 'p15':[],'p16':[],'p17':[],'p18':[],'p19':[],'p20':[],'p21':[],'p22':[],'p23':[],'p24':[],'p25':[],'p26':[]}
expo1_pp1_Tph   = {'p1':[], 'p2':[], 'p3':[], 'p4':[], 'p5':[], 'p6':[], 'p7':[], 'p8':[], 'p9':[], 'p10':[], 'p11':[], 'p12':[], 'p13':[], 'p14':[], 'p15':[],'p16':[],'p17':[],'p18':[],'p19':[],'p20':[],'p21':[],'p22':[],'p23':[],'p24':[],'p25':[],'p26':[]}
expo1_pp1_Tau   = {'p1':[], 'p2':[], 'p3':[], 'p4':[], 'p5':[], 'p6':[], 'p7':[], 'p8':[], 'p9':[], 'p10':[], 'p11':[], 'p12':[], 'p13':[], 'p14':[], 'p15':[],'p16':[],'p17':[],'p18':[],'p19':[],'p20':[],'p21':[],'p22':[],'p23':[],'p24':[],'p25':[],'p26':[]}
expo1_pp2_Tbar  = {'p1':[], 'p2':[], 'p3':[], 'p4':[], 'p5':[], 'p6':[], 'p7':[], 'p8':[], 'p9':[], 'p10':[], 'p11':[], 'p12':[], 'p13':[], 'p14':[], 'p15':[],'p16':[],'p17':[],'p18':[],'p19':[],'p20':[],'p21':[],'p22':[],'p23':[],'p24':[],'p25':[],'p26':[]}
expo1_pp2_Tph   = {'p1':[], 'p2':[], 'p3':[], 'p4':[], 'p5':[], 'p6':[], 'p7':[], 'p8':[], 'p9':[], 'p10':[], 'p11':[], 'p12':[], 'p13':[], 'p14':[], 'p15':[],'p16':[],'p17':[],'p18':[],'p19':[],'p20':[],'p21':[],'p22':[],'p23':[],'p24':[],'p25':[],'p26':[]}
expo1_pp2_Tau   = {'p1':[], 'p2':[], 'p3':[], 'p4':[], 'p5':[], 'p6':[], 'p7':[], 'p8':[], 'p9':[], 'p10':[], 'p11':[], 'p12':[], 'p13':[], 'p14':[], 'p15':[],'p16':[],'p17':[],'p18':[],'p19':[],'p20':[],'p21':[],'p22':[],'p23':[],'p24':[],'p25':[],'p26':[]}


for i,k in zip(p_txt.keys(), p_n):
    for n,m in zip(p_x_heat[i], p_y_heat_pp2[i]):
            if n >= x0[i]:
                p_x_exp1[i].append(n)
                p_y_exp1[i].append(m)
            if x0[i] <= n <x0[i][0]+20:    
                p_x_exp1_20s[i].append(n)
                p_y_exp1_20s[i].append(m)
            p_x_exp1[i].insert(0,x0[i][0])
            p_y_exp1[i].insert(0,0)
            p_x_exp1_20s[i].insert(0,x0[i][0])
            p_y_exp1_20s[i].insert(0,0)            
    p_x_exp1[i] = [A - p_x_exp1[i][0] for A in p_x_exp1[i]]
    p_x_exp1_20s[i] = [A - p_x_exp1_20s[i][0] for A in p_x_exp1_20s[i]]

    #Curve fitting expo1 - pp1
    def Sat_pp1(x1, a1, b1, c1):
            return a1*(1 - np.exp(-(x1-c1)/b1))
    p0_1 = (0.1, 1.4, 1)
    params_1, cv_1 = optimize.curve_fit(Sat_pp1, p_x_exp1[i], p_y_exp1[i], p0_1)
    a1, b1, c1 = params_1
    coef_a1[i].append(a1)
    coef_b1[i].append(b1)
    coef_c1[i].append(c1)
    expo1_pp1_Tbar[i].append(a1)
    expo1_pp1_Tph[i].append(a1/b1) 
    expo1_pp1_Tau[i].append(b1)
    # print(i, '   Tp_heat = ', expo1_pp1_Tph[i], '\t Tbar_expo1 = ', expo1_pp1_Tbar[i], '\t Tau_expo1 = ', expo1_pp1_Tau[i] )

    #Curve fitting expo1 - pp2 - 20s
    def Sat_pp2(x2, a2, b2, c2):
            return a2*(1 - np.exp(-(x2-c2)/b2))
    p0_2 = (0.1, 1.4, 1)
    params_2, cv_2 = optimize.curve_fit(Sat_pp2, p_x_exp1_20s[i], p_y_exp1_20s[i], p0_2)
    a2, b2, c2 = params_2
    coef_a2[i].append(a2)
    coef_b2[i].append(b2)
    coef_c2[i].append(c2)
    expo1_pp2_Tbar[i].append(a2)
    expo1_pp2_Tph[i].append(a2/b2) 
    expo1_pp2_Tau[i].append(b2)
    # print(i, '   Tp_heat = ', expo1_pp2_Tph[i], '\t Tbar_expo1 = ', expo1_pp2_Tbar[i], '\t Tau_expo1 = ', expo1_pp2_Tau[i] )
 

#----------------------------------------------------  GRAPH EXP01 ----------------------------------------------------#
for n in p_n:                                
    fig = plt.figure(n)
    ax = fig.add_subplot(111)
    plt.title(filename + r" - Level "+ str(n) + r" - $\sigma_{max}$ = " + str(contrainte[n-1]) +'MPa_EXPO_1')
    plt.grid()
    plt.plot(p_x_exp1['p'+str(n)], p_y_exp1['p'+str(n)], 'navy', label="data")
    plt.plot(p_x_exp1['p'+str(n)], Sat_pp1(p_x_exp1['p'+str(n)], coef_a1['p'+str(n)][0], coef_b1['p'+str(n)][0], coef_c1['p'+str(n)][0]), 'y--', label ="fitted")
    plt.plot(p_x_exp1_20s['p'+str(n)], Sat_pp2(p_x_exp1_20s['p'+str(n)], coef_a2['p'+str(n)][0], coef_b2['p'+str(n)][0], coef_c2['p'+str(n)][0]), 'r--', label ="fitted_20s")
    plt.xlabel('Time, 'r"$t$" ' (s)')
    plt.ylabel('Temperature elevation, 'r'$\theta$' ' (°C)')
    plt.legend()
    textstr = '\n'.join((r'$f= 30Hz$',r'$R = 10 $',r'$T= 10°C$'))
    ax.text(0.85, 0.85, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
    #plt.savefig(filename + '/' + filename + ' - Level '+ str(n) + ' - Sigma_max_ ' + str(contrainte[n-1]) +'MPa.pdf')
    #plt.savefig(filename + '/' + filename + ' - Level '+ str(n) + ' - Sigma_max_ ' + str(contrainte[n-1]) +'MPa.png')
plt.show()

#----------------------------------------------------  Identification - EXPO 2 ----------------------------------------------------#
print('\n2.3    Exponential Curve n°2')
x00 = {'p1':[100.28], 'p2':[100.12], 'p3':[100.14], 'p4':[100.19], 'p5':[100.20], 'p6':[100.20], 'p7':[100.20], 'p8':[100.20], 'p9':[100.20], 'p10':[100.30], 'p11':[100.50], 'p12':[100.40], 'p13':[100.60], 'p14':[100.70], 'p15':[100.80],'p16':[100.70],'p17':[100.70],'p18':[100.80],'p19':[100.90],'p20':[100.80],'p21':[101.00],'p22':[101.2],'p23':[101.30],'p24':[101.40],'p25':[101.50]}
p_x_exp2        = {'p1':[], 'p2':[], 'p3':[], 'p4':[], 'p5':[], 'p6':[], 'p7':[], 'p8':[], 'p9':[], 'p10':[], 'p11':[], 'p12':[], 'p13':[], 'p14':[], 'p15':[],'p16':[],'p17':[],'p18':[],'p19':[],'p20':[],'p21':[],'p22':[],'p23':[],'p24':[],'p25':[]}
p_y_exp2        = {'p1':[], 'p2':[], 'p3':[], 'p4':[], 'p5':[], 'p6':[], 'p7':[], 'p8':[], 'p9':[], 'p10':[], 'p11':[], 'p12':[], 'p13':[], 'p14':[], 'p15':[],'p16':[],'p17':[],'p18':[],'p19':[],'p20':[],'p21':[],'p22':[],'p23':[],'p24':[],'p25':[]}
p_x_exp2_end    = {'p1':[], 'p2':[], 'p3':[], 'p4':[], 'p5':[], 'p6':[], 'p7':[], 'p8':[], 'p9':[], 'p10':[], 'p11':[], 'p12':[], 'p13':[], 'p14':[], 'p15':[],'p16':[],'p17':[],'p18':[],'p19':[],'p20':[],'p21':[],'p22':[],'p23':[],'p24':[],'p25':[]}
p_y_exp2_end    = {'p1':[], 'p2':[], 'p3':[], 'p4':[], 'p5':[], 'p6':[], 'p7':[], 'p8':[], 'p9':[], 'p10':[], 'p11':[], 'p12':[], 'p13':[], 'p14':[], 'p15':[],'p16':[],'p17':[],'p18':[],'p19':[],'p20':[],'p21':[],'p22':[],'p23':[],'p24':[],'p25':[]}

coef_start_e1   = {'p1':[], 'p2':[], 'p3':[], 'p4':[], 'p5':[], 'p6':[], 'p7':[], 'p8':[], 'p9':[], 'p10':[], 'p11':[], 'p12':[], 'p13':[], 'p14':[], 'p15':[],'p16':[],'p17':[],'p18':[],'p19':[],'p20':[],'p21':[],'p22':[],'p23':[],'p24':[],'p25':[]}
coef_start_e2   = {'p1':[], 'p2':[], 'p3':[], 'p4':[], 'p5':[], 'p6':[], 'p7':[], 'p8':[], 'p9':[], 'p10':[], 'p11':[], 'p12':[], 'p13':[], 'p14':[], 'p15':[],'p16':[],'p17':[],'p18':[],'p19':[],'p20':[],'p21':[],'p22':[],'p23':[],'p24':[],'p25':[]}
coef_start_e3   = {'p1':[], 'p2':[], 'p3':[], 'p4':[], 'p5':[], 'p6':[], 'p7':[], 'p8':[], 'p9':[], 'p10':[], 'p11':[], 'p12':[], 'p13':[], 'p14':[], 'p15':[],'p16':[],'p17':[],'p18':[],'p19':[],'p20':[],'p21':[],'p22':[],'p23':[],'p24':[],'p25':[]}
coef_start_e4   = {'p1':[], 'p2':[], 'p3':[], 'p4':[], 'p5':[], 'p6':[], 'p7':[], 'p8':[], 'p9':[], 'p10':[], 'p11':[], 'p12':[], 'p13':[], 'p14':[], 'p15':[],'p16':[],'p17':[],'p18':[],'p19':[],'p20':[],'p21':[],'p22':[],'p23':[],'p24':[],'p25':[]}
coef_end_e1     = {'p1':[], 'p2':[], 'p3':[], 'p4':[], 'p5':[], 'p6':[], 'p7':[], 'p8':[], 'p9':[], 'p10':[], 'p11':[], 'p12':[], 'p13':[], 'p14':[], 'p15':[],'p16':[],'p17':[],'p18':[],'p19':[],'p20':[],'p21':[],'p22':[],'p23':[],'p24':[],'p25':[]}
coef_end_e2     = {'p1':[], 'p2':[], 'p3':[], 'p4':[], 'p5':[], 'p6':[], 'p7':[], 'p8':[], 'p9':[], 'p10':[], 'p11':[], 'p12':[], 'p13':[], 'p14':[], 'p15':[],'p16':[],'p17':[],'p18':[],'p19':[],'p20':[],'p21':[],'p22':[],'p23':[],'p24':[],'p25':[]}
coef_end_e3     = {'p1':[], 'p2':[], 'p3':[], 'p4':[], 'p5':[], 'p6':[], 'p7':[], 'p8':[], 'p9':[], 'p10':[], 'p11':[], 'p12':[], 'p13':[], 'p14':[], 'p15':[],'p16':[],'p17':[],'p18':[],'p19':[],'p20':[],'p21':[],'p22':[],'p23':[],'p24':[],'p25':[]}
coef_end_e4     = {'p1':[], 'p2':[], 'p3':[], 'p4':[], 'p5':[], 'p6':[], 'p7':[], 'p8':[], 'p9':[], 'p10':[], 'p11':[], 'p12':[], 'p13':[], 'p14':[], 'p15':[],'p16':[],'p17':[],'p18':[],'p19':[],'p20':[],'p21':[],'p22':[],'p23':[],'p24':[],'p25':[]}

expo2_pp1_Tpc_1   = {'p1':[], 'p2':[], 'p3':[], 'p4':[], 'p5':[], 'p6':[], 'p7':[], 'p8':[], 'p9':[], 'p10':[], 'p11':[], 'p12':[], 'p13':[], 'p14':[], 'p15':[],'p16':[],'p17':[],'p18':[],'p19':[],'p20':[],'p21':[],'p22':[],'p23':[],'p24':[],'p25':[]}
expo2_pp1_Tpc_2   = {'p1':[], 'p2':[], 'p3':[], 'p4':[], 'p5':[], 'p6':[], 'p7':[], 'p8':[], 'p9':[], 'p10':[], 'p11':[], 'p12':[], 'p13':[], 'p14':[], 'p15':[],'p16':[],'p17':[],'p18':[],'p19':[],'p20':[],'p21':[],'p22':[],'p23':[],'p24':[],'p25':[]}
expo2_pp1_Tau   = {'p1':[], 'p2':[], 'p3':[], 'p4':[], 'p5':[], 'p6':[], 'p7':[], 'p8':[], 'p9':[], 'p10':[], 'p11':[], 'p12':[], 'p13':[], 'p14':[], 'p15':[],'p16':[],'p17':[],'p18':[],'p19':[],'p20':[],'p21':[],'p22':[],'p23':[],'p24':[],'p25':[]}
expo2_pp2_Tau   = {'p1':[], 'p2':[], 'p3':[], 'p4':[], 'p5':[], 'p6':[], 'p7':[], 'p8':[], 'p9':[], 'p10':[], 'p11':[], 'p12':[], 'p13':[], 'p14':[], 'p15':[],'p16':[],'p17':[],'p18':[],'p19':[],'p20':[],'p21':[],'p22':[],'p23':[],'p24':[],'p25':[]}
Rc              = {'p1':[], 'p2':[], 'p3':[], 'p4':[], 'p5':[], 'p6':[], 'p7':[], 'p8':[], 'p9':[], 'p10':[], 'p11':[], 'p12':[], 'p13':[], 'p14':[], 'p15':[],'p16':[],'p17':[],'p18':[],'p19':[],'p20':[],'p21':[],'p22':[],'p23':[],'p24':[],'p25':[]}

for i,k in zip(p_txt.keys(), p_n):                                     
    for n,m in zip(p_x_cool[i], p_y_cool[i]):
        if x00[i] <= n < x00[i][0]+8  :
            p_x_exp2[i].append(n)
            p_y_exp2[i].append(m)
        if x00[i][0]+8 <= n < x00[i][0]+80:
            p_x_exp2_end[i].append(n)
            p_y_exp2_end[i].append(m)           
  
    p_x_exp2[i] = [A - (p_x_exp2[i][0]) for A in p_x_exp2[i]]
    p_x_exp2_end[i] = [A - p_x_exp2_end[i][0]+p_x_exp2[i][-1] for A in p_x_exp2_end[i]]

    #Curve fitting expo 2 - start - pp1
    def Expo2_pp1(x1_1, e1_1, e2_1, e3_1, e4_1):
        return e1_1 * np.exp(-(x1_1 - e2_1)/e3_1) + e4_1

    p02 = (expo1_pp1_Tbar[i][0],0.01, 1.3, 0.025)
    params2, cv2 = optimize.curve_fit(Expo2_pp1, p_x_exp2[i], p_y_exp2[i], p02)
    e1_1, e2_1, e3_1, e4_1 = params2
    coef_start_e1[i].append(e1_1)
    coef_start_e2[i].append(e2_1)
    coef_start_e3[i].append(e3_1)
    coef_start_e4[i].append(e4_1)

    # determine quality of the fit
    squaredDiffs = np.square(p_y_exp2[i] - Expo2_pp1(p_x_exp2[i], e1_1, e2_1, e3_1, e4_1))
    squaredDiffsFromMean = np.square(p_y_exp2[i] - np.mean(p_y_exp2[i]))
    rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
    
    expo2_pp1_Tau[i].append(e3_1)
    expo2_pp1_Tpc_1[i].append((expo1_pp1_Tbar[i][0] - e4_1) / e3_1)
    expo2_pp1_Tpc_2[i].append((expo1_pp2_Tbar[i][0] - e4_1) / e3_1)
    Rc[i].append(rSquared)
    # print(i, '\t Tp_cool_1 = ', expo2_pp1_Tpc_1[i][0], '\t Tp_cool_2 = ', expo2_pp1_Tpc_2[i][0], '\t Tau eq = ', expo2_pp1_Tau[i][0], '\t R² = ', Rc[i][0])

    #Curve fitting expo 2 - end - pp2
    def Expo2_pp2(x1_2, e1_2, e2_2, e3_2, e4_2):
        return e1_2 * np.exp(-(x1_2 - e2_2)/e3_2) + e4_2

    p02_2 = (expo1_pp1_Tbar[i][0],0.01, 1.3, 0.025)
    params2_2, cv2_2 = optimize.curve_fit(Expo2_pp2, p_x_exp2_end[i], p_y_exp2_end[i], p02_2)
    e1_2, e2_2, e3_2, e4_2 = params2_2
    coef_end_e1[i].append(e1_2)
    coef_end_e2[i].append(e2_2)
    coef_end_e3[i].append(e3_2)
    coef_end_e4[i].append(e4_2)
    expo2_pp2_Tau[i].append(e3_2)
#----------------------------------------------------  GRAPH EXP02 ----------------------------------------------------#
# for n in p_n:                                
#     fig = plt.figure(n)
#     ax = fig.add_subplot(111)
#     plt.title(filename + r" - Level "+ str(n) + r" - $\sigma_{max}$ = " + str(contrainte[n-1]) +'MPa_EXPO_2')
#     plt.grid()
#     plt.plot(p_x_exp2['p'+str(n)], p_y_exp2['p'+str(n)], 'navy', label="exp data")
#     plt.plot(p_x_exp2['p'+str(n)], Expo2_pp1(p_x_exp2['p'+str(n)], coef_start_e1['p'+str(n)][0], coef_start_e2['p'+str(n)][0], coef_start_e3['p'+str(n)][0], coef_start_e4['p'+str(n)][0]), 'y--', label =r"fitted - $Tau_{eq} = %.3f$ s"%(coef_start_e3['p'+str(n)][0]))
#     plt.plot(p_x_exp2_end['p'+str(n)], p_y_exp2_end['p'+str(n)], 'navy')
#     plt.plot(p_x_exp2_end['p'+str(n)], Expo2_pp2(p_x_exp2_end['p'+str(n)], coef_end_e1['p'+str(n)][0], coef_end_e2['p'+str(n)][0], coef_end_e3['p'+str(n)][0], coef_end_e4['p'+str(n)][0]), 'r--', label =r"fitted - $Tau_{eq} = %.3f$ s"%(coef_end_e3['p'+str(n)][0]))

#     # plt.plot([85, 100],[Theta_bar_end['p'+str(n)], Theta_bar_end['p'+str(n)]], 'r', label ='Temperature elevation, 'r"$\bar{\theta}^{0D}$")
#     plt.xlabel('Time, 'r"$t$" ' (s)')
#     plt.ylabel('Temperature elevation, 'r'$\theta$' ' (°C)')
#     plt.legend()
#     textstr = '\n'.join((r'$f= 30Hz$',r'$R = 10 $',r'$T= 10°C$'))
#     ax.text(0.08, 0.90, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
#     #plt.savefig(filename + '/' + filename + ' - Level '+ str(n) + ' - Sigma_max_ ' + str(contrainte[n-1]) +'MPa.pdf')
#     #plt.savefig(filename + '/' + filename + ' - Level '+ str(n) + ' - Sigma_max_ ' + str(contrainte[n-1]) +'MPa.png')
# plt.show()

#----------------------------------------------------  Identification - Harmonic ----------------------------------------------------#
# print('\n2.4    Harmonic')
# # A travailler avec les transformées de Fourier
# for i in p_txt.keys():                                    
#         Cmax = max(p_y_end[i])
#         Cmin = min(p_y_end[i])
#         couplage_ampl[i].append(Cmax - Cmin)
#         #print(i, couplage_ampl[i][0])


#----------------------------------------------------  Self-heating curves ----------------------------------------------------#
print('\n3.    Courbe AE')
Y_power_1 = []
Y_power_2 = []
coef_A3 = {'p1':[], 'p2':[], 'p3':[], 'p4':[], 'p5':[], 'p6':[], 'p7':[], 'p8':[], 'p9':[], 'p10':[], 'p11':[], 'p12':[], 'p13':[], 'p14':[], 'p15':[],'p16':[],'p17':[],'p18':[],'p19':[],'p20':[],'p21':[],'p22':[],'p23':[],'p24':[],'p25':[]}
coef_m3 = {'p1':[], 'p2':[], 'p3':[], 'p4':[], 'p5':[], 'p6':[], 'p7':[], 'p8':[], 'p9':[], 'p10':[], 'p11':[], 'p12':[], 'p13':[], 'p14':[], 'p15':[],'p16':[],'p17':[],'p18':[],'p19':[],'p20':[],'p21':[],'p22':[],'p23':[],'p24':[],'p25':[]}

for n,k in zip(p_txt, p_n):
    if 0 < k <=25:          #stabilisation jusqu'à k=21
        Y_power_1.append(expo1_pp1_Tbar[n][0])
        Y_power_2.append(expo1_pp2_Tbar[n][0]) 

def Power(x3, A3, m3):
    return A3*(pow(x3,m3))

#----------------------------------------------------  GRAPH - Self-heating curves ----------------------------------------------------#
# fig = plt.figure('SH curves', figsize = [15, 8])
# ax1 = fig.add_subplot(121)
# plt.grid(b=True, which='major', linestyle='-')
# plt.grid(b=True, which='minor', axis= 'x', linestyle='--')
# plt.title(filename + ' Self-Heating Curve')
# plt.plot(contrainte[0:25] ,Y_power_1, '.b', markersize=10, label ='Tbar_pp1_fitting 150cycles')
# plt.plot(contrainte[0:25] ,Y_power_2, '.r', markersize=10, label ='Tbar_pp2_fitting 3000cycles')
# #plt.plot(contrainte, Power(contrainte, A3, m3), 'k--', label = "fitted curve")
# #plt.text(100, 4.5, "$y = %.1E * x^{%.3f} $"%(A3, m3), size = 15)
# plt.ylabel(r"Temperature elevation, $\bar{\theta}^{0D}$ (°C)")
# plt.xlabel(r"Maximum compressive stress, $\sigma_{max}$ (MPa)")
# plt.legend()
# textstr = '\n'.join((r'$f= 30Hz$',r'$R = 10 $',r'$T= 10°C$'))
# ax1.text(0.08, 0.90, textstr, transform=ax1.transAxes, fontsize=10, verticalalignment='top', bbox=props)

# ax2 = fig.add_subplot(122)
# plt.title(filename + ' Self-Heating Curve (Log-Log)')
# plt.loglog(contrainte[0:25] ,Y_power_1, '.b', markersize=10, label ='Tbar_pp1_fitting 150cycles')
# plt.loglog(contrainte[0:25] ,Y_power_2, '.r', markersize=10, label ='Tbar_pp2_fitting 3000cycles')
# #plt.loglog(contrainte, Power(contrainte, A3, m3), 'k--', label = "fitted curve")
# #plt.text(100, 4.5, "$y = %.1E * x^{%.3f} $"%(A3, m3), size = 15)
# plt.ylabel(r"Temperature elevation, $\bar{\theta}^{0D}$ (°C)")
# plt.xlabel(r"Maximum compressive stress, $\sigma_{max}$ (MPa)")
# plt.grid(b=True, which='major', linestyle='-')
# plt.grid(b=True, which='minor', axis= 'x', linestyle='--')
# plt.legend()
# textstr = '\n'.join((r'$f= 30Hz$',r'$R = 10 $',r'$T= 10°C$'))
# ax2.text(0.08, 0.90, textstr, transform=ax2.transAxes, fontsize=10, verticalalignment='top', bbox=props)
# plt.show()

#----------------------------------------------------  Multiples check ----------------------------------------------------#
print('\n4.    Multiple checks')
SH_V1 = []
SH_V2 = []
SH_V3 = []
SH_V4 = []
SH_V5 = []
SH_V6 = []
SH_V7 = []
SH_V8 = []
SH_V9 = []
SH_V10 = []
SH_V11 = []
SH_V12 = []
for k,n in zip(p_txt, p_n):
    if 0 < n <=25:          #on garde seulement jusqu'a k=21
        SH_V1.append(expo1_pp1_Tph[k][0])
        SH_V2.append(expo1_pp2_Tph[k][0])
        SH_V3.append(expo1_pp1_Tbar[k][0] / expo2_pp1_Tau[k][0])
        SH_V4.append(expo1_pp2_Tbar[k][0] / expo2_pp1_Tau[k][0])
        SH_V5.append(expo2_pp1_Tpc_1[k][0])
        SH_V6.append(expo2_pp1_Tpc_2[k][0])
        SH_V7.append(expo1_pp1_Tau[k][0])
        SH_V8.append(expo1_pp2_Tau[k][0])
        SH_V9.append(expo2_pp1_Tau[k][0])
        SH_V10.append(expo2_pp2_Tau[k][0])

# ----------------------------------------------------  GRAPH - Multiples check ----------------------------------------------------#
# fig4 = plt.figure('SH - Multiple checks')
# ax1 = fig4.add_subplot(111)
# plt.title(filename + ' SH curves - checks')
# plt.grid(axis='both')
# plt.plot(contrainte[0:25] ,SH_V1[0:25], '.', markersize=10, color='r', label = r"$\dot{\theta}^{heating}$ with pp1 (3000 cylces)")
# plt.plot(contrainte[0:25] ,SH_V2[0:25], '^', markersize=5, color='r', label = r"$\dot{\theta}^{heating}$ with pp2 (150 cycles)")
# plt.plot(contrainte[0:25] ,SH_V3[0:25], '.', markersize=10, color='g', label = r"$\bar{\theta} / \tau_{eq}$ with pp1 (3000 cylces)")
# plt.plot(contrainte[0:25] ,SH_V4[0:25], '^', markersize=5, color='g', label = r"$\bar{\theta} / \tau_{eq}$ with pp2 (150 cylces)")
# plt.plot(contrainte[0:25] ,SH_V5[0:25], '.', markersize=10, color='b', label = r"$\dot{\theta}^{cooling}$ with pp1 (3000 cylces)")
# plt.plot(contrainte[0:25] ,SH_V6[0:25], '^', markersize=5, color='b', label = r"$\dot{\theta}^{cooling}$ with pp1 (150 cylces)")
# plt.xlabel(r"Compressive Stress, $\sigma_{max}$ (MPa)")
# plt.ylabel(r"Temperature elevation rate, (°C/s)")
# plt.legend(loc='best', fontsize = 15)
# textstr = '\n'.join((r'$f= 30Hz$',r'$R = 10 $',r'$T= 10°C$'))
# ax1.text(0.90, 0.10, textstr, transform=ax1.transAxes, fontsize=10, verticalalignment='top', bbox=props)
# # ax2.legend(loc='upper right', fontsize = 15)
# # plt.show()

#----------------------------------------------------  Tau eq study ----------------------------------------------------#
print('\n5.    Tau_eq')

# avg_tau_eq_AE1 = statistics.mean(V5[4:15])
# std_tau_eq_AE1 = statistics.stdev(V5[4:15])
# med_tau_eq_AE1 = statistics.median(V5[4:15])
# avg_tau_eq_AE2 = statistics.mean(V5[15:22])  
# std_tau_eq_AE2 = statistics.stdev(V5[15:22])
# med_tau_eq_AE2 = statistics.median(V5[15:22])

# fig5 = plt.figure('SH - Tau')
# ax1 = fig5.add_subplot(111)
# plt.title(filename + ' SH curves - checks')
# plt.grid(axis='both')
# plt.plot(contrainte[0:25], SH_V7[0:25] ,'.',markersize = 10, label = 'expo1_pp1_Tau')
# plt.plot(contrainte[0:25], SH_V8[0:25] ,'^',markersize = 5, label = 'expo1_pp2_Tau')
# plt.plot(contrainte[0:25], SH_V9[0:25] ,'.',markersize = 10, label = 'expo2_pp1_Tau')
# plt.plot(contrainte[0:25], SH_V10[0:25],'^',markersize = 5, label = 'expo2_pp2_Tau')
# plt.legend()
# # ax1.plot(Y_power[0:15] ,V5[0:15],'*b', markersize=10, label = r"$\tau{eq}$"+ ' - AE1' + ' -' +"$Avg = %.3f $"%(avg_tau_eq_AE1)+ "$ - Med = %.3f $"%(med_tau_eq_AE1)+ "$ - Std = %.3f $"%(std_tau_eq_AE1))
# # ax1.plot(Y_power[15:22] ,V5[15:22],'*r', markersize=10, label = r"$\tau{eq}$" + ' - AE2'+ ' - ' +"$Avg = %.3f $"%(avg_tau_eq_AE2)+ "$ - Med = %.3f $"%(med_tau_eq_AE2)+ "$ - Std = %.3f $"%(std_tau_eq_AE2))
# plt.xlabel(r"Temperature elevation, $\bar{\theta}^{0D}$ (°C)")
# plt.ylabel( r"$\tau{eq}$ (s)")


plt.show()