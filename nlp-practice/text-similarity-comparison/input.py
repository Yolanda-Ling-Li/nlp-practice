import os
import shutil



thedata_dir = os.path.abspath(os.path.join(os.path.curdir, "thedata"))
input_dir = os.path.abspath(os.path.join(os.path.curdir, "data\\input"))
if not os.path.exists(thedata_dir):
    print("###!!Wrong: thedata folder is not detected!###")
if not os.path.exists(input_dir):
    os.makedirs(input_dir)

one_path = os.path.join(thedata_dir, "txt\\gaigo_txt")
for f in os.listdir(one_path):
    shutil.copyfile(one_path + "\\" + f, input_dir + "\\" + f)

two_path = os.path.join(thedata_dir, "txt\\internet_txt")
for f in os.listdir(two_path):
    shutil.copyfile(two_path + "\\" + f, input_dir + "\\" + f)
