from PIL import Image
import glob
import os

in_dir = "./data/org_img/akiyosi_Green"
out_dir = "./data/doukutsu_img"

test_path = './data/org_img/cave01/gen_start_scale=0/0.png'

dir_path = glob.glob(in_dir+"/*")

for i, f in enumerate(dir_path):
    print(f'aa:{i} bb{f}')

