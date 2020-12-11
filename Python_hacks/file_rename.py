# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 14:48:24 2020

@author: Pierre Massat <pmassat@stanford.edu>

Important note: Windows has a path size limit of 260 characters, when adding the path plus filename. Hence if a path is too long, os.rename will throw a FileNotFoundError (WinError 3).
"""

import os, glob
from datetime import date

today = date.today()
other_date = date(2020,7,29)

os.chdir(r'C:\Users\Pierre\Desktop\Postdoc\TmVO4\TmVO4_heat-capacity\2017-07_TmVO4_Cp_MCE\2017-07-28--31\Massaic_MCE\Run2_0p5uA_figures\2020-12_MCE_fitting')

filenames = glob.glob('* *')# create list of filenames containing at least one whitespace

for filename in filenames:
    print(filename)
    # newfname = '_'.join([str(today), 'DyVO4_ECE', filename.replace(' ', '_')])
    newfname = filename.replace(' ', '')
    # New file name adds today's date at the beginning and replaces whitespaces with underscores
    print(newfname)
    print()
    # os.rename(filename, newfname)

