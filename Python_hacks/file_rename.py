# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 14:48:24 2020

@author: Pierre Massat <pmassat@stanford.edu>
"""

import os, glob
from datetime import date

today = date.today()
other_date = date(2020,3,19)

# os.chdir(r'C:\Users\Pierre\Desktop\Postdoc\Writing\Articles\2020_TmVO4_model_nematicity\Ian-s_edits')
# os.chdir(r'C:\Users\Pierre\Desktop\Postdoc\TmVO4\TmVO4_NMR\2020-04_TmVO4_NMR')
os.chdir(r'C:\Users\Pierre\Documents\Personnel\Famille\Famille_Massat')

filenames = glob.glob('* *')# create list of filenames containing at least one whitespace

for filename in filenames:
    print(filename)
    newfname = '_'.join([str(today), filename.replace(' ', '_')])
    # New file name adds today's date at the beginning and replaces whitespaces with underscores
    print(newfname)
    print()
    # os.rename(filename, newfname)

