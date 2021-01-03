import os
import PIL
from   PIL import Image

GT = 'C:\\Users\\Cédric\\Desktop\\Test Data Sets\\Middlebury\\GT'
RESIZED = 'C:\\Users\\Cédric\\Desktop\\Test Data Sets\\Middlebury\\RESIZED'

for dir in os.listdir(GT):
    print('-'*50)
    print(dir)
    print('-'*50)
    dir_path_gt = os.path.join(GT, dir)
    dir_path_resized = os.path.join(RESIZED, dir)
    if not os.path.exists(dir_path_resized):
        os.mkdir(dir_path_resized)
    
    for subdir in os.listdir(dir_path_gt):
        subdir_path_gt = os.path.join(dir_path_gt, subdir)
        subdir_path_resized = os.path.join(dir_path_resized, subdir)
        if not os.path.exists(subdir_path_resized):
            os.mkdir(subdir_path_resized)
            
        frames = os.listdir(subdir_path_gt)
        input_frames = [frames[0], frames[-1]]
        for img in input_frames:
            im_path_gt = os.path.join(subdir_path_gt, img)
            im_path_resized = os.path.join(subdir_path_resized, img)
            
            img = Image.open(im_path_gt)
            w, h = img.size
            img_smol = img.resize((w // 4, h // 4), PIL.Image.LANCZOS)
            
            img_smol.save(im_path_resized)
        
        print(subdir)
            
            