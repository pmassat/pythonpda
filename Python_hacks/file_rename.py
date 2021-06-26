# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 14:48:24 2020

@author: Pierre Massat <pmassat@stanford.edu>

Important note: Windows has a path size limit of 260 characters, when adding the path plus filename. Hence if a path is too long, os.rename will throw a FileNotFoundError (WinError 3).
"""

import os, glob
from datetime import date

today = date.today()
other_date = date(2021,6,24)

os.chdir(r'C:\Users\Pierre\Desktop\Postdoc\Technical_stuff\PPMS\PPMS_DR\2021_DR_return')

filenames = glob.glob('* *.pdf')# create list of filenames containing at least one whitespace

for idx, filename in enumerate(filenames):
    print(filename)
    newfname = '_'.join([str(today), filename.replace(' ', '_')])
    # newfname = filename.replace('_compressed.JPG', '.JPG')
    # New file name adds today's date at the beginning and replaces whitespaces with underscores
    print(newfname)
    print()
    # os.rename(filename, newfname)

