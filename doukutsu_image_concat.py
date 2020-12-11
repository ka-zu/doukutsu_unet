from PIL import Image
import glob
import sys

"""
研究用
分割した画像をくっつけて元の画像にするプログラム
"""

def concatH(im1, im2):
    img = Image.new('RGB', (im1.width + im2.width, im1.height))
    img.paste(im1, (0, 0))
    img.paste(im2, (im1.width, 0))
    return img

def concatV(im1, im2):
    img = Image.new('RGB', (im1.width, im1.height + im2.height))
    img.paste(im1, (0, 0))
    img.paste(im2, (0, im1.height))
    return img


#引数指定したときのメイン関数
if __name__=='__main__' and len(sys.argv) == 4:
    args = sys.argv
    print(f"引数　width:{args[1]} height:{args[2]} size:{args[3]}")

    img_paths = glob.glob("./data/doukutsu_test/CIMG1250*")
    
    #print(img_paths)
    print(len(img_paths))
    #print(sorted(img_paths))

    #縦の枚数
    height_num = args[1]
    #横の枚数
    width_num = args[2]
    #画像サイズ
    img_size = args[3]

    """
    full_img = Image.new('RGB', (width_num*img_size, height_num*img_size))
    print(full_img.size)
    #im = Image.open(f'./data/doukutsu_test/CIMG1250_1_1.jpg').show()
    
    for w in range(width_num+1):
        for h in range(height_num+1):
            im = Image.open(f'./data/doukutsu_test/CIMG1250_{w}_{h}.jpg')
            full_img.paste(im, (w*img_size, h*img_size))
    
    full_img.save('./result.jpg')
    """

#引数指定しなかったときのメイン関数
elif __name__ == '__main__':
    img_paths = glob.glob("./data/doukutsu_test/CIMG1250*")
    
    print(img_paths)
    print(len(img_paths))
    #print(sorted(img_paths))

    #縦の枚数
    height_num = 25
    #横の枚数
    width_num = 34
    #画像サイズ
    img_size = 128

    """
    full_img = Image.new('RGB', (width_num*img_size, height_num*img_size))
    print(full_img.size)
    #im = Image.open(f'./data/doukutsu_test/CIMG1250_1_1.jpg').show()
    
    for w in range(width_num+1):
        for h in range(height_num+1):
            im = Image.open(f'./data/doukutsu_test/CIMG1250_{w}_{h}.jpg')
            full_img.paste(im, (w*img_size, h*img_size))
    
    full_img.save('./result.jpg')
    """
        
