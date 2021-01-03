import argparse
import cv2
from   metrics import metrics
from   metrics.LPIPSmodels import util
import metrics.LPIPSmodels.dist_model as dm
import numpy
import os
import pandas as pd
from   utils import tools

'''
Adapted from https://github.com/thunil/TecoGAN/blob/master/metrics.py
'''

parser = argparse.ArgumentParser(description = 'Compute PSNR, SSIM, LPIPS, tOF and tLP for two directories')
parser.add_argument('--target', type = str, default = None, help = 'Directory containing targets')
parser.add_argument('--result', type = str, default = None, help = 'Directory containing results')
parser.add_argument('--output', type = str, default = None, help = 'Directory for metrics output')
parser.add_argument('--id', type = str, default = 'metrics', help = 'ID for csv output')
parser.add_argument('--version', type = str, default = '0.1', help = 'Version of LPIPS net to use (default: 0.1)')
parser.add_argument('--no_gpu', action = 'store_true', help = 'Don\'t use the GPU')
parser.set_defaults(no_gpu = False)

args = parser.parse_args()

assert args.target != None and args.result != None

if not os.path.exists(args.output):
    os.mkdir(args.output)

with open(os.path.join(args.output, 'arguments.txt'), 'w') as f:
    for k,v in sorted(vars(args).items()): print('{0}: {1}'.format(k,v), file = f)

result_list = os.listdir(args.result)
target_list = os.listdir(args.target)
folder_n = len(target_list)

model = dm.DistModel()
model.initialize(model = 'net-lin', net = 'alex', use_gpu = not args.no_gpu)

cutfr = 0#2
maxV = 0.4 #, for line 154-166

keys = ['PSNR', 'SSIM', 'LPIPS', 'tOF', 'tLP100']
sum_dict    = dict.fromkeys(['FrameAvg_'+_ for _ in keys], 0)
len_dict    = dict.fromkeys(keys, 0)
avg_dict    = dict.fromkeys(['Avg_'+_ for _ in keys], 0)
folder_dict = dict.fromkeys(['FolderAvg_'+_ for _ in keys], 0)

for folder_i in range(folder_n):
    result = tools.listIMGinDIR(os.path.join(args.result, result_list[folder_i]))
    target = tools.listIMGinDIR(os.path.join(args.target, target_list[folder_i]))
    image_no = len(target)

    print('-'*50)
    print(result_list[folder_i])

    list_dict = {}
    for key_i in keys:
        list_dict[key_i] = []
    
    for i in range(cutfr, image_no-cutfr):
        output_img = cv2.imread(result[i])[:,:,::-1]
        target_img = cv2.imread(target[i])[:,:,::-1]
        msg = 'frame %d, tar %s, out %s, '%(i, str(target_img.shape), str(output_img.shape))
        if( target_img.shape[0] < output_img.shape[0]) or ( target_img.shape[1] < output_img.shape[1]): # target is not dividable by 4
            output_img = output_img[:target_img.shape[0],:target_img.shape[1]]
        # print(result[i])
        
        if 'tOF' in keys: # tOF
            output_grey = cv2.cvtColor(output_img, cv2.COLOR_RGB2GRAY)
            target_grey = cv2.cvtColor(target_img, cv2.COLOR_RGB2GRAY)
            if (i > cutfr): # temporal metrics
                target_OF=cv2.calcOpticalFlowFarneback(pre_tar_grey, target_grey, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                output_OF=cv2.calcOpticalFlowFarneback(pre_out_grey, output_grey, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                target_OF, ofy, ofx = metrics.crop_8x8(target_OF)
                output_OF, ofy, ofx = metrics.crop_8x8(output_OF)
                OF_diff = numpy.absolute(target_OF - output_OF)
                if False: # for motion visualization
                    tOFpath = os.path.join(args.output,'%03d_tOF'%folder_i)
                    if(not os.path.exists(tOFpath)): os.mkdir(tOFpath)
                    hsv = numpy.zeros_like(output_img)
                    hsv[...,1] = 255
                    out_path = os.path.join(tOFpath, 'flow_%04d.jpg' %i)
                    mag, ang = cv2.cartToPolar(OF_diff[...,0], OF_diff[...,1])
                    # print('tar max %02.6f, min %02.6f, avg %02.6f' % (mag.max(), mag.min(), mag.mean()))
                    mag = numpy.clip(mag, 0.0, maxV)/maxV
                    hsv[...,0] = ang*180/numpy.pi/2
                    hsv[...,2] = mag * 255.0 #
                    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
                    cv2.imwrite(out_path, bgr)
                    
                OF_diff = numpy.sqrt(numpy.sum(OF_diff * OF_diff, axis = -1)) # l1 vector norm
                # OF_diff, ofy, ofx = crop_8x8(OF_diff)
                list_dict['tOF'].append( OF_diff.mean() )
                msg += 'tOF %02.2f, ' %(list_dict['tOF'][-1])
            
            pre_out_grey = output_grey
            pre_tar_grey = target_grey

        target_img, ofy, ofx = metrics.crop_8x8(target_img)
        output_img, ofy, ofx = metrics.crop_8x8(output_img)
            
        if 'PSNR' in keys:# psnr
            list_dict['PSNR'].append(metrics.psnr(target_img, output_img))
            msg +='psnr %02.2f' %(list_dict['PSNR'][-1])
        
        if 'SSIM' in keys:# ssim
            list_dict['SSIM'].append(metrics.ssim(target_img, output_img))
            msg +=', ssim %02.2f' %(list_dict['SSIM'][-1])
            
        if 'LPIPS' in keys or 'tLP100' in keys:
            img0 = metrics.im2tensor(target_img) # RGB image from [-1,1]
            img1 = metrics.im2tensor(output_img)
            if not args.no_gpu:
                img0 = img0.cuda()
                img1 = img1.cuda()
        
            if 'LPIPS' in keys: # LPIPS
                dist01 = model.forward(img0, img1)
                list_dict['LPIPS'].append(dist01[0])
                msg +=', lpips %02.2f' %(dist01[0])
            
            if 'tLP100' in keys and (i > cutfr):# tLP, temporal metrics
                dist0t = model.forward(pre_img0, img0)
                dist1t = model.forward(pre_img1, img1)
                # print ('tardis %f, outdis %f' %(dist0t, dist1t))
                dist01t = numpy.absolute(dist0t - dist1t) * 100.0 ##########!!!!!
                list_dict['tLP100'].append( dist01t[0] )
                msg += ', tLPx100 %02.2f' %(dist01t[0])
            pre_img0 = img0
            pre_img1 = img1
        
        msg += ', crop (%d, %d)' %(ofy, ofx)
        print(msg)
    mode = 'w' if folder_i==0 else 'a'
    
    pd_dict = {}
    for cur_num_data in keys:
        num_data = cur_num_data+'_%02d' % folder_i
        cur_list = numpy.float32(list_dict[cur_num_data])
        pd_dict[num_data] = pd.Series(cur_list)

        num_data_sum = cur_list.sum()
        num_data_len = cur_list.shape[0]
        num_data_mean = num_data_sum / num_data_len
        print('%s, max %02.4f, min %02.4f, avg %02.4f' % 
            (num_data, cur_list.max(), cur_list.min(), num_data_mean))
            
        if folder_i == 0:
            avg_dict['Avg_' + cur_num_data] = [num_data_mean]
        else:
            avg_dict['Avg_' + cur_num_data] += [num_data_mean]
        
        sum_dict['FrameAvg_' + cur_num_data] += num_data_sum
        len_dict[cur_num_data] += num_data_len
        folder_dict['FolderAvg_' + cur_num_data] += num_data_mean
        
    pd.DataFrame(pd_dict).to_csv(os.path.join(args.output, 'metrics.csv'), mode=mode)
    
for num_data in keys:
    sum_dict['FrameAvg_' + num_data] = pd.Series([sum_dict['FrameAvg_' + num_data] / len_dict[num_data]])
    folder_dict['FolderAvg_' + num_data] = pd.Series([folder_dict['FolderAvg_' + num_data] / folder_n])
    avg_dict['Avg_' + num_data] = pd.Series(numpy.float32(avg_dict['Avg_'+num_data]))
    print('%s, total frame %d, total avg %02.4f, folder avg %02.4f' % 
        (num_data, len_dict[num_data], sum_dict['FrameAvg_' + num_data][0], folder_dict['FolderAvg_' + num_data][0]))

pd.DataFrame(avg_dict).to_csv(os.path.join(args.output, args.id + '.csv'), mode='a')
pd.DataFrame(folder_dict).to_csv(os.path.join(args.output, args.id + '.csv'), mode='a')
pd.DataFrame(sum_dict).to_csv(os.path.join(args.output, args.id + '.csv'), mode='a')
print('Finished.')