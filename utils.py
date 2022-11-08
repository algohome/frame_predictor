import numpy as np
from PIL import Image, ImageOps
import glob
import os
import sys
import pathlib


def log(outname, **kwargs):
    if not outname.endswith('.txt'):
        outname += '.txt'
    with open(outname, 'w') as outfile:
        outfile.write(outname+'\n'+'*'*20+'\n')
        for kw in kwargs:
            outfile.write(str(kw) + ': ' + str(kwargs[kw])+'\n')
    
def make_square(im, min_size=512, fill_color=(0, 0, 0, 0)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im

def check_or_make_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        
def split_and_show(arr, denorm=False):
        for i in range(arr.shape[-1]//3):
            img = arr[0,:,:,3*i : 3*(i+1)]
            if denorm:
                img *= 255
            Image.fromarray((img).astype('uint8')).show()
        