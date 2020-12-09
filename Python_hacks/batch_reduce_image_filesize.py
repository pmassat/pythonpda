# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 19:11:47 2020

@author: Pierre Massat <pmassat@stanford.edu>

Batch reduce image sizes.
Code inside for loop adapted from https://stackoverflow.com/a/13211834/3460390
"""

import os
import glob
from datetime import date
from PIL import Image

today = date.today()
other_date = date(2020, 7, 29)

os.chdir(r'C:\Users\Pierre\Desktop\Postdoc\YTmVO4\YTmVO4_pictures\YVO4\YVO4-LS5259\YVO4-LS5259-Cij\YVO4-LS5259-Cij-4')

# create list of filenames containing at least one whitespace
filenames = glob.glob('* - Copy.png')

for filename in filenames:
    foo = Image.open(filename)
    print(foo.size)
    # New file name replaces whitespaces with underscores
    newfname = [filename.replace(' - ', '_')]
    newfname.append(newfname[-1].replace('.png', '.jpg'))
    print(newfname[-1])
    # Downsize the image with an ANTIALIAS filter (gives the highest quality)
    # foo = foo.resize((160,300),Image.ANTIALIAS)
    # foo.save("path\\to\\save\\image_scaled.jpg",quality=95)
    foo.save(newfname[-1], optimize=True, quality=50)
