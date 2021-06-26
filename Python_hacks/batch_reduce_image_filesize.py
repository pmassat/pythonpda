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

os.chdir(r'C:\Users\Pierre\Desktop\Postdoc\YTmVO4\YTmVO4_pictures\Y99-TmVO4-LS5677\Y99-TmVO4-LS5677-Cij2106')

# create list of filenames containing at least one whitespace
filenames = glob.glob('* - Copy.png')

for filename in filenames:
    print(f'{os.path.getsize(filename)/1024**2:.2f} MB')

    # New file name replaces whitespaces with underscores
    newfname = filename.replace(' - Copy.png', '_copy.jpg')
    print(newfname)

    foo = Image.open(filename)
    print(foo.size, '\n')
    # Downsize the image with an ANTIALIAS filter (gives the highest quality)
    # if os.path.getsize(filename)/1024**2>1:
    #     foo.save(newfname, optimize=True, quality=50)
        # foo = foo.resize(tuple(int(x/2) for x in foo.size),Image.ANTIALIAS)
        # foo.save("path\\to\\save\\image_scaled.jpg",quality=95)
