import numpy
import os
import sys
import threading
import torch
import torch.nn as nn
from   torch.autograd import gradcheck, Variable

def l1_loss(output, target):
    loss = [ (t-o)*(t-o) for o, t in zip(output, target) ]
    return torch.sum(torch.stack(loss))


def charbonnier_loss(output, target, epsilon = 10e-6):
    loss = [ torch.mean(torch.sqrt((t-o)*(t-o) + epsilon*epsilon)) for o, t in zip(output, target) ]
    return torch.sum(torch.stack(loss))


def negPSNR_loss(x,epsilon):
    loss = torch.mean(torch.mean(torch.mean(torch.sqrt(x * x + epsilon * epsilon),dim=1),dim=1),dim=1)
    return torch.mean(-torch.log(1.0/loss) /100.0)


def tv_loss(x,epsilon):
    loss = torch.mean( torch.sqrt(
        (x[:, :, :-1, :-1] - x[:, :, 1:, :-1]) ** 2 +
        (x[:, :, :-1, :-1] - x[:, :, :-1, 1:]) ** 2 + epsilon *epsilon
            )
        )
    return loss

    
def gra_adap_tv_loss(flow, image, epsilon):
    w = torch.exp( - torch.sum(	torch.abs(image[:,:,:-1, :-1] - image[:,:,1:, :-1]) + 
                                torch.abs(image[:,:,:-1, :-1] - image[:,:,:-1, 1:]), dim = 1))		
    tv = torch.sum(torch.sqrt((flow[:, :, :-1, :-1] - flow[:, :, 1:, :-1]) ** 2 + (flow[:, :, :-1, :-1] - flow[:, :, :-1, 1:]) ** 2 + epsilon *epsilon) ,dim=1)             
    loss = torch.mean( w * tv )
    return loss	
        

def smooth_loss(x,epsilon):
    loss = torch.mean(
        torch.sqrt(
            (x[:,:,:-1,:-1] - x[:,:,1:,:-1]) **2 +
            (x[:,:,:-1,:-1] - x[:,:,:-1,1:]) **2 + epsilon**2
        )
    )
    return loss
    
    
def motion_sym_loss(offset, epsilon, occlusion = None):
    if occlusion == None:
        return torch.mean(torch.sqrt((offset[0] + offset[1])**2 + epsilon **2))
    else:
        return torch.mean(torch.sqrt((offset[0] + offset[1])**2 + epsilon **2))


def part_loss(diffs, offsets, images, epsilon, use_negPSNR=False):
    if use_negPSNR:
        loss_pixel = [ 
            torch.sum(torch.stack([ negPSNR_loss(diff, epsilon) for diff in diffs[0] ])), 
            torch.sum(torch.stack([ negPSNR_loss(diff, epsilon) for diff in diffs[1] ])) 
            ]
    else:
        loss_pixel = [ 
            torch.sum(torch.stack([ charbonier_loss(diff, epsilon) for diff in diffs[0] ])), 
            torch.sum(torch.stack([ charbonier_loss(diff, epsilon) for diff in diffs[1] ])) 
            ]

    if offsets[0][0] is not None:
        loss_offset = [ torch.sum(torch.stack([
                gra_adap_tv_loss(off_0, image, epsilon) + gra_adap_tv_loss(off_1, images[-1], epsilon) for off_0, off_1, image in zip(offsets[0], offsets[1], images[:-1])
                ])) ]
    else:
        loss_offset = [ Variable(torch.zeros(1).cuda()) ]

    loss_sym = [ torch.sum(torch.stack([ motion_sym_loss([off_0, off_1], epsilon=epsilon) for off_0, off_1 in zip(offsets[0], offsets[1]) ])) ]
    
    return loss_pixel, loss_offset, loss_sym


'''
in_1 and in_2 expected to be in the same order for TecoGAN Ping-Pong Loss
'''
def pingpong_loss(in_1, in_2):
    return torch.mean(torch.stack([ torch.abs(x-y) for x,y in zip(in_1, in_2) ]))

'''
BCE loss
'''
def gan_loss(score_real, score_fake, score_gen, epsilon, batchsize = 0):
    loss_real = torch.log(score_real + epsilon)
    loss_fake = torch.log(1 - score_fake + epsilon)
    loss_dis  = -(loss_real + loss_fake)
    loss_gen  = -torch.log(score_gen + epsilon)

    return torch.mean(loss_gen), torch.mean(loss_dis), torch.mean(loss_real).detach(), torch.mean(loss_fake).detach()


def layerloss(layer_1, layer_2):
    fix_range = 0.02
    layer_norm = [12.0, 14.0, 24.0, 100.0]
    layer_loss = 0

    for l1, l2, n in zip(layer_1, layer_2, layer_norm):
        diff = torch.abs(l1 - l2)
        loss = torch.mean(torch.sum(diff, dim = 1))
        loss_scaled = fix_range * loss / n
        layer_loss += loss_scaled

    return layer_loss


'''
Input: two lists of tensors
Output: tensor with sum of cosine similarity
'''
def cosine_similarity(t1, t2, w = None):
    weights = [ 1.0 for t in t1 ] if w == None else w
    out = 0
    for x, y, w in zip(t1, t2, weights):
        c = torch.sum(x*y, dim = 1)
        c = 1.0 - torch.mean(c)
        out += w*c
    return out