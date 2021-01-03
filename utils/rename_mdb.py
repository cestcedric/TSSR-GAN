import os
import shutil

root = 'C:\\Users\\Cédric\\Desktop\\testdata\\Middlebury'
for d1 in os.listdir(root):
    p1 = os.path.join(root, d1)
    
    for d2 in os.listdir(p1):
        if d2 == 'septuplets':
            old = ['frame07.png', 'frame08.png', 'frame09.png', 'frame10.png', 'frame11.png', 'frame12.png', 'frame13.png']
            new = ['0.png', '1.png', '2.png', '3.png', '4.png', '5.png', '6.png']
        if d2 == 'triplets':
            old = ['frame10.png', 'frame10i11.png', 'frame11.png']
            new = ['0.png', '1.png', '2.png']
        p2 = os.path.join(p1, d2)
        
        for d3 in os.listdir(p2):
            p3 = os.path.join(p2, d3)
            
            for o, n in zip(old, new):
                try:
                    shutil.move(os.path.join(p3, o), os.path.join(p3, n))
                except:
                    continue
            print(p3)
                
            
            
            