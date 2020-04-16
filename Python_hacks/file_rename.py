# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 14:48:24 2020

@author: Pierre Massat <pmassat@stanford.edu>
"""

import os

os.chdir(r'C:\Users\Pierre\Desktop\Postdoc\Writing\Articles\2020_TmVO4_model_nematicity\Ian-s_edits')

mystr = "2020-04-14_First two paragraphs of the results section.docx"

os.rename(mystr, mystr.replace(' ', '_'))

