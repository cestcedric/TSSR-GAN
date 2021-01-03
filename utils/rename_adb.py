import os
import shutil

root = 'C:\\Users\\Cédric\\Desktop\\Test Data Sets\\Adobe240'
old = ['0.jpg', '3.jpg', '6.jpg']
new = ['0.jpg', '1.jpg', '2.jpg']

for d1 in os.listdir(root):
    p1 = os.path.join(root, d1)
    
    for d2 in os.listdir(p1):
        if d2 == 'septuplets':
            continue
        p2 = os.path.join(p1, d2)
        
        for d3 in os.listdir(p2):
            p3 = os.path.join(p2, d3)
            
            for o, n in zip(old, new):
                shutil.move(os.path.join(p3, o), os.path.join(p3, n))
            print(p3)
                
            
            
            