from PIL import Image
import glob
import os



def trim_img128(dir_name, img_path):
    im = Image.open(img_path)
    im = im.convert('RGB')

    img_name = img_path.split("/")[5].split(".")[0]
    print("name ="+img_name)

    save_path = out_dir +"/"
    print(save_path)

    if not os.path.exists(save_path):
        os.mkdir(save_path)
        
    print(save_path +dir_name +"_"+ img_name + "_1.jpg")
    
    x, y = im.size()
    
    print(f'x:{x}')
    #左上
    im_crop = im.crop((0,0,  128,128)).save(save_path +dir_name +"_"+ img_name + "_1.jpg")
    #右上
    im_crop = im.crop((251-128,0,  251,128)).save(save_path +dir_name +"_"+ img_name + "_2.jpg")
    #左下
    im_crop = im.crop((0,188-128,  128,188)).save(save_path +dir_name +"_"+ img_name + "_3.jpg")
    #右下
    im_crop = im.crop((251-128,188-128,  251,188)).save(save_path +dir_name +"_"+ img_name + "_4.jpg")
    #真ん中
    img_w, img_h = im.size
    im_crop = im.crop(((img_w - 128)//2,
                       (img_h - 128)//2,
                       (img_w + 128)//2,
                       (img_h + 128)//2)).save(save_path +dir_name +"_"+ img_name + "_5.jpg")
    
def trim_img(img_path, size):
    
    out_dir = "./data/doukutsu_test"
    
    im = Image.open(img_path)
    im = im.convert('RGB')

    img_name = img_path.split("\\")[1].split(".")[0]
    
    #print("name = "+img_name)
    
    save_path = out_dir +"/"
    print(save_path)

    if not os.path.exists(save_path):
        os.mkdir(save_path)
        
    print("save:"+save_path + img_name+ "_" + str(1) + "_"+str(1)+".jpg")
    
    x, y = im.size
    
    print(f'x:{x} y:{y}')
    
    n = x//size
    m = y//size
    
    print(f'n:{n} m:{m}')
    
    for N in range(n-1):
        for M in range(m-1):
            im.crop((128*N, 128*M,  128*(N+1), 128*(M+1))).save(save_path + img_name + "_"+ str(N) + "_"+str(M)+".jpg")
    
    
    
if __name__ == "__main__":
    
    #trim_img128("aaa", test_path)

    
    in_dir = "./data/org_img/akiyosi_test"
    
    files = glob.glob(in_dir+"/*")
    
    print(files)
    
    for file in files:
        trim_img(file, 128)
        
    
    """
    in_dir = "./data/org_img"
    out_dir = "./data/doukutsu_img"

    test_path = './data/org_img/cave01/gen_start_scale=0/0.png'

    dir_path = glob.glob(in_dir+"/*")

    print(dir_path)
    
    
    for dir_item in dir_path:
        print(dir_item)
        dir_name = dir_item.split("/")[3]
        print(dir_name)
        img_path = glob.glob(str(dir_item) + "/gen_start_scale=0/*")
        print(img_path)
        
        for img_item in img_path:
            print(img_item)
            trim_img128(dir_name, img_item)
    """
