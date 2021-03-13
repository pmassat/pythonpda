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

os.chdir(r'C:\Users\Pierre\Desktop\2021-03-11_phone_pics')

# create list of filenames containing at least one whitespace
filenames = glob.glob('*.jpg')

for filename in filenames:
    foo = Image.open(filename)
    print(foo.size)
    print(os.path.getsize(filename)/1024**2)

    # New file name replaces whitespaces with underscores
    newfname = [filename.replace('_compressed.JPG', '.JPG')]
    # newfname.append(newfname[-1].replace('.png', '.jpg'))
    print(newfname[-1], '\n')

    # Downsize the image with an ANTIALIAS filter (gives the highest quality)
    # if os.path.getsize(filename)/1024**2>1:
        # foo = foo.resize(tuple(int(x/2) for x in foo.size),Image.ANTIALIAS)
        # foo.save("path\\to\\save\\image_scaled.jpg",quality=95)
        # foo.save(newfname[-1], optimize=True, quality=50)
