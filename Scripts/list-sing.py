import glob
import os
import numpy as np
import sys

current_dir = "C:/Users/HiepNguyen1304/Music/darknet-master/data/test"
file_train = open("data/test.txt", "w")
for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.jpg")):  
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        file_train.write(current_dir + "/" + title + '.jpg' + "\n")
file_train.close()
