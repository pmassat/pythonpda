# -*- coding: utf-8 -*-
"""
Created on Wed May 20 11:58:55 2020

@author: Pierre Massat <pmassat@stanford.edu>
"""

#%% Import modules
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import convolve2d as conv2

from average_Cp import averageCpwithH2
from Cp_fit_functions import Cp_TFIM, Cp_LFIM

#%% Matplotlib settings

#%% Data importation
sample = 'TmVO4-RF-E'
filename = 'TmVO4_RF-E_2017-07-14.dat'
os.chdir(r'C:\Users\Pierre\Desktop\Postdoc\TmVO4\TmVO4_heat-capacity\2017-07_TmVO4_Cp_MCE\2017-07-20_Cp')
Data = pd.read_csv(filename, skiprows=list(range(13))+[14], encoding = "ISO-8859-1")
# Encoding required to avoir "UnicodeDecodeError: 'utf-8' codec can't decode byte 0xb5 in position 12: invalid start byte"
os.chdir("./2017-07-20_TmVO4_Cp_analysis")

#%% Sample properties
m = 0.25e-3# mass of sample, in g
M = 283.87# molar mass of TmVO4, in g/mol
Tc0 = 2.126# Value of transition temperature in this sample

#%% Create variables
Hc0 = 0.51# value in Tesla units of the critical field at zero temperature
# in the absence of demagnetizing factor
# see Data taken on needles of TmVO4-LS5200 in July 2017
rescaling = Hc0/0.69# rescaling factor, due to demag 
# obtained from fit using normal PDF of fields close to Hc see below
H = Data["Field (Oersted)"]*rescaling
T = Data["Sample Temp (Kelvin)"]
Cp = Data["Samp HC (J/mole-K)"]
CpErr = Data["Samp HC Err (J/mole-K)"]

whichPoints = np.isfinite(H) & np.isfinite(T) & np.isfinite(Cp)
H=H[whichPoints]
T=T[whichPoints]
Cp=Cp[whichPoints]*M/m*1e-6# factor 1e-6 converts from uJ/K to J/K
CpErr=CpErr[whichPoints]*M/m*1e-6#
uh1 = np.unique(np.around(H,decimals=1))
uh = np.copy(uh1[1:])
uh[0] = uh1[0]# When rounding to nearest tens, remove uh = 10 Oe because it is redundant with uh = 0 Oe

#%% plot 2D scatter of Cp Data at H=0
plt.figure()
plt.plot(T[np.round(H,-1)<20], Cp[np.round(H,-1)<20], '.')
# xlim([0 hmax])ylim([0 tmax])
plt.xlabel('Temperature (K)')
plt.ylabel('Cp (J/K/mol)')

#%% Gaussian convolution
tstep = 0.025 
hstep = 500*rescaling
x = tstep*np.arange(-50,50)
s = 0.05
d1Gaussian = -np.exp(-x**2/(2*s**2))*x/np.sqrt(s**6*2*np.pi)
d2Gaussian = np.exp(-x**2/(2*s**2))*(x**2 - s**2)/np.sqrt(s**10*2*np.pi)
d1Cp = np.convolve(Cp, d1Gaussian,'same')

#%% Separate Data according to value of field
separatedCpData = [None]*len(uh)
# cols = ['H', 'T', 'Cp', 'd1Cp', 'CpErr']
for i in range(len(uh)):
    wp = np.abs(H-uh[i])<50
    split_by_field = pd.DataFrame()
    split_by_field['H'] = H[wp]
    split_by_field['T'] = T[wp]
    split_by_field['Cp'] = Cp[wp]
    split_by_field['d1Cp'] = d1Cp[wp]
    split_by_field['CpErr'] = CpErr[wp]
    
    separatedCpData[i] = split_by_field.sort_values(by='T')
    # Warning: do not use .T to reference the temperature column of 
    # separatedCpData[i], since .T is also the notation for matrix transposition!
    
#%% Average Datapoints taken at any given temperature and field setpoint
# We want to compute the average of Data points that are repetitions of the
# same measurement, i.e. with same temperature and field setpoints
avgData = [None]*len(uh)
R = 8.314# Gas constant, in J/K/mol
Tsep = 2e-3# Data points taken within an interval of Tsep are considered to be measured at the same temperature setpoint
Tp = 27# Temperature scale of phonons contribution to Cp in TmVO4, in K see 'TmVO4_Cp_phonons.m'
for i in range(len(uh)):
    avgData[i] = averageCpwithH2(Tsep,separatedCpData[i]['T'],separatedCpData[i].Cp, 
                                 separatedCpData[i].CpErr,separatedCpData[i].H)

    avgData[i]['Cpel'] = avgData[i].Cp - R*(avgData[i]['T']/Tp)**3# electronic contribution to Cp, after subtracting phonons contribution
    avgData[i]['Cpelr'] = avgData[i].Cpel/R

#%% plot averaged Data 
i = 1
plt.figure()
plt.plot(avgData[i]['T'], avgData[i].Cp, '.', label='$C_p^{\mathrm{full}}$')
plt.plot(avgData[i]['T'], R*(avgData[i]['T']/Tp)**3, '.', label='$C_p^{\mathrm{phonons}}$')
plt.plot(avgData[i]['T'], avgData[i].Cpel, '.', label='$C_p^{\mathrm{full}}-C_p^{\mathrm{phonons}}$')
plt.legend()
plt.title('TmVO$_4$ heat capacity')
plt.xlabel('T (K)') 
plt.ylabel('$C_p$ (J/K/mol)')

#%% Identify experimental critical temperature at each field
# Then we can plot Tc vs uh and fit using equation f(h) = Tc/Hc*h/atanh(h/Hc)
# From this, we can see that the experimental value of Hc is ~0.72T 
# and hence correct for demag, knowing the value of critical field Hc0 (see beginning of code)
M = ones(1,length(uh))
Tcd1 = ones(1,length(uh))
for i=1:length(uh)
    [M[i],I] = min(avgData[i].d1Cp)
    Tcd1[i] = avgData[i]['T'][i]
end

#%% Prepare MF fit of Cp vs Temperature at given field
j=1
Tfit = avgData[j]['T']
Cpfit = avgData[j].Cp
CpfitErr = avgData[j].CpFullErr
wghts = 1./CpfitErr
Tmaxfit = 2.15

#%% plt.plot averaged Data at each field separately
plt.figure()
# for i=rng
#     fp = fplot(@(t)Cp_TFIM(t/Tc0,uh[i]/(Hc0*1e4)),[0 maxTplot],'LineWidth',2)
#     fp = fplot(@(t)Cp_TFIM_offset_strain(t/2.125,uh[i]/(5.1e3),1.5e-3),[0 4],'LineWidth',2)
# Fit parameters on Data at H=0: Tc=2.125(3), e=1.5(4)e-3
# Note: the values of amplitude coefficient and Tc extracted from fit 
# in curve fitting tool using Cp_TFIM (no offset strain) are A=7.35 and Tc=2.142K
for i in range(0, 9, 2):
    plot_label = abs(uh[i]/(Hc0*1e4))
    plt.errorbar(avgData[i]['T'], avgData[i].Cpelr, avgData[i].CpFullErr/R,
                 marker='o', linewidth=0, elinewidth=2, 
                 label=f"{plot_label:.2g}")
    # plt.plot(avgData[i]['T'], avgData[i].Cpelr)
plt.xlabel('Temperature (K)') 
plt.ylabel('C$_p$/R')#plt.ylabel('C$_p$ (JK$^{-1}$mol$^{-1}$)')
plt.legend(title="$H/H_c$")
# plt.xlim([0, maxTplot])# maxTplot = 3.2#
# plt.title('Heat capacity of TmVO4 at various fields')

#%% Prepare fit of Cp vs Temperature 
i = 9
# Use these variables in Curve fitting tool
clear fitT fitCp fitCpErr fitwghts 
fitT = avgData[i]['T']
fitCp = avgData[i].Cpelr
fitCpErr = avgData[i].CpFullErr/R
fitwghts = 1./fitCpErr
# Tc = 2.15

#%% Compute Cp for Gaussian distribution of fields
Cptheo = [None]*len(uh)
for i in range(len(uh)):
    Cptheo[i] = {}
    Cptheo[i]['h'] = uh[i]/(Hc0*1e4)
    Cptheo[i]['rhsgm'] = 0.09
    Cptheo[i]['sgm'] = Cptheo[i]['h']*Cptheo[i]['rhsgm']
    Cptheo[i]['T_single_h'] = np.linspace(2.5e-3, 1.5, 601)# reduced temperature T/Tc
    Cptheo[i]['single_h'] = np.zeros(Cptheo[i]['T_single_h'].shape)
    Cptheo[i]['T_phenomeno'] = np.linspace(5e-3, 1.5, 300)# reduced temperature T/Tc
    Cptheo[i]['phenomeno'] = np.zeros(Cptheo[i]['T_phenomeno'].shape)

i = 0
Cptheo[i]['single_h'] = Cp_TFIM(Cptheo[i]['T_single_h'], Cptheo[i]['h'])
Cptheo[i]['phenomeno'] = Cp_LFIM(Cptheo[i]['T_phenomeno'], 1.0e-3)
# Fit parameters on Data at H=0: Tc=2.125(3), e=1.5(4)e-3
# Cp_LFIM(h=0)
for i in range(2, 9, 2):
    Cptheo[i]['single_h'] = Cp_TFIM(Cptheo[i]['T_single_h'],Cptheo[i]['h'])
    # for j in range(1,len(Cptheo[i]['T_phenomeno'])):
    #     Cptheo[i]['phenomeno'][j] = CpTFIM_normpdf(Cptheo[i]['T_phenomeno'][j], 
    #                                                Cptheo[i]['h'],Cptheo[i].sgm)

#%% plot Cp for Gaussian distribution of fields
i=0
plt.figure()
plt.plot(avgData[i]['T'], avgData[i].Cpelr, '.', label='Data')
h_label = f"h={Cptheo[i]['h']:.2g}"
plt.plot(Cptheo[i]['T_single_h']*Tc0, Cptheo[i]['single_h'], label=h_label)
plt.plot(Cptheo[i]['T_phenomeno']*Tc0,Cptheo[i]['phenomeno'],
         label=f"{h_label},r={Cptheo[i]['rhsgm']:.1e}")
# plt.title(['Cp$\_$TFIM vs CpTFIM$\_$normpdf' f" H=%.0fOe",uh[i])])
# lgd = plt.legend() plt.title(lgd,'TmVO4-RF-E')

#%% plot averaged Data at each field separately
plt.figure() hold on
clr = lines(length(rng))
eb = cell(size(rng))
for i=rng
# fp = fplot(@(t)Cp_TFIM(t/Tc0,Cptheo[i]['h']),[0 3.2],'--','LineWidth',2,'Color',clr(rng==i,:))
plt.plot(Cptheo[i]['T_single_h']*Tc0,Cptheo[i]['single_h'], '--','Color',clr(rng==i,:), label=sprintf('h=%.2f',Cptheo[i]['h']))
plt.plot(Cptheo[i]['T_phenomeno']*Tc0,Cptheo[i]['phenomeno'],'Color',clr(rng==i,:), label=...
    sprintf('h=%.2f,r=%.1e',Cptheo[i]['h'],Cptheo[i]['rhsgm']))
end
for i=rng
eb{rng==i} = plt.errorbar(avgData[i]['T'],avgData[i].Cpelr,avgData[i].CpFullErr/R,...
    '.','MarkerSize',18, label=num2str(uh[i]/(Hc0*1e4),'%.2f'),...
    'Color',clr(rng==i,:),'LineWidth',2)
end
plt.xlabel('$T$ (K)') plt.ylabel('$C_p/R$')#plt.ylabel('C$_p$ (JK$^{-1}$mol$^{-1}$)')
xlim([0 max(Cptheo(rng(1))['T_single_h']*Tc0)])
lgd = plt.legend([eb{:}]) lgd['T']itle.String = '$H/H_c$'
ax = gca ax.YMinorTick = 'on'# Add minor ticks on Y axis
anntheo = annotation('textbox',[0.13 0.83 0.2 0.1],'interpreter','latex',...
    'String',{['$--$ 1-parameter fit'] ['----- 2-parameter fit']...
    }, 'LineStyle','-','EdgeColor','k',...
    'FitBoxToText','on','LineWidth',1,'BackgroundColor','w','Color','k')# add annotation
annnum = annotation('textbox',[0.01 0.01 0.2 0.1],'interpreter','latex',...
    'String',{['(a)']}, 'LineStyle','-','EdgeColor','None',...
    'BackgroundColor','none','Color','k','VerticalAlignment','bottom')# add numbering annotation
# plt.title('Heat capacity of TmVO$_4$')
grid on#
hold off

#%% Compute additional Cp for Gaussian distribution of fields
# Cptheo[i]['h'] = 0.87
Cptheo[i]['rhsgm'] = 0.1
Cptheo[i].sgm = Cptheo[i]['h']*Cptheo[i]['rhsgm']
for j=1:length(Cptheo[i]['T'])
    Cptheo[i]['phenomeno'][j] = CpTFIM_normpdf(Cptheo[i]['T'][j],Cptheo[i]['h'],Cptheo[i].sgm)
end
plt.plot(Cptheo[i]['T']*Tc0,Cptheo[i]['phenomeno'], label=sprintf('h=%.2f,r=%.1e',Cptheo[i]['h'],Cptheo[i]['rhsgm']))

#%% plt.plot additional Cp for Gaussian distribution of fields
offset = 0.01
plt.plot(Cptheo[i]['T']*Tc0,Cptheo[i]['phenomeno']+offset, label=sprintf('h=%.2f,r=%.1e',Cptheo[i]['h'],Cptheo[i]['rhsgm']))

#%% Prepare MF fit of Cp vs Temperature under field
index = 15
H1 = uh(index)
T1 = avgData(index)['T']
Cp1 = avgData(index).Cp/R
Cp1Err = avgData(index).CpFullErr/R
wghts1 = 1./Cp1Err

#%% Fit and plt.plot
maxTfit = 2#Kelvin
hrstr = sprintf('%.2f',H1/(Hc0*1e4))
# [fitresult1, gof1] = fitCpTFIM(T1,Cp1,wghts1,2.142,H1/5.1e3)
[fitresult1, gof1] = fitSchTemp(T1,Cp1,wghts1,maxTfit)
fitprms = coeffvalues(fitresult1)
prmerr = coeffvalues(fitresult1)-confint(fitresult1)
gapstr = [sprintf('%.2f',fitprms(1)) '$\pm$' sprintf('%.2f',prmerr(1,1))]
offsetstr = [sprintf('%.3f',fitprms(2)) '$\pm$' sprintf('%.3f',prmerr(1,2))]
plt.title({['Schottky anomaly fit of ' sample],[' at $H/H_c=$' hrstr]})
annfit = annotation('textbox',[0.575 0.175 0.2 0.1],'interpreter','latex',...
    'String',{['$R**2=$ ' sprintf('%.4f',gof1.rsquare)] ['$\Delta=$ ' gapstr 'K']...
    ['offset ' offsetstr]}, 'LineStyle','-','EdgeColor','k',...
    'FitBoxToText','on','LineWidth',1,'BackgroundColor','w','Color','k')# add annotation
formatplt.figure()
annfit.Position(2)=.175

#%% Export plt.figure() to pdf
# formatplt.figure()
printPDF([todaystr '_TmVO4-RF-E_Cp_fit'])
# printPDF(['2019-06-18_TmVO4-RF-E_fit_Schottky_' strrep(hrstr,'.','p') 'xHc'])



#%% 3D scatter of Cp(T) at each field separately
plt.figure()
for i=1:length(uh)
    scatter3(separatedCpData[i].H,separatedCpData[i]['T'],separatedCpData[i].Cp,'.',...
        'Displayname',num2str(uh[i]', '%d Oe'))
#     plt.plot(separatedCpData[i]['T'],separatedCpData[i].Cp,'-+')
    hold on
end
plt.xlabel('H (Oe)')plt.ylabel('T (K)')zlabel('C_p (\muJ/K)')
xlim([0 hmax])
plt.legend()
hold off

#%% plt.plot dCp/dT at each field separately
# Need to average Data points measured at each temperature first

plt.figure()
for i=1:length(uh)
    scatter3(separatedCpData[i].H,separatedCpData[i]['T'],-separatedCpData[i].d1Cp,'.',...
        'Displayname',num2str(uh[i]', '%d Oe'))
#     plt.plot(separatedCpData[i]['T'],separatedCpData[i].Cp,'-+')
    hold on
end
plt.xlabel('H (Oe)')plt.ylabel('T (K)')zlabel('-dC_p/dT (\muJ/K)')
xlim([0 hmax])
plt.legend()
hold off

#%% plt.plot Cp(T) at each field separately
#%%
plt.figure()
for i=1:length(uh)
    plt.plot(separatedCpData[i]['T'],separatedCpData[i].Cp,'-+')
    hold on
end
plt.xlabel('Temperature (K)')
plt.ylabel('Cp/T (J/K**2/mol)')
plt.legendCell = cellstr(num2str(uh, '%-d Oe'))
plt.legend(plt.legendCell)
hold off
