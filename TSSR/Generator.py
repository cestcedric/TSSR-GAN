from   my_package.DepthFlowProjection import DepthFlowProjectionModule
from   my_package.FilterInterpolation import FilterInterpolationModule
from   my_package.FlowProjection import FlowProjectionModule
import numpy
import random
import torch
import torch.nn as nn
from   torch.autograd.variable import Variable
from   submodules.Upscaling.Utility import pad_if_divide, upsample
from   utils import tools
from   utils.Stack import Stack

import submodules.AHDRNet.model as AHDRNet
from   submodules import MegaDepth
from   submodules import PWCNet
from   submodules import Resblock
from   submodules import S2D_models
from  .Upscaler import Composer

class Generator(torch.nn.Module):
    def __init__(self,
                 channel = 3,
                 scale = 4,
                 filter_size = 4,
                 training = False,
                 improved_estimation = 2,
                 detach_estimation = False,
                 interpolated_frames = 1):

        # base class initialization
        super(Generator, self).__init__()
        
        self.filter_size = filter_size
        self.training = training
        self.interpolated_frames = interpolated_frames
        self.ie_threshold = improved_estimation * 128
        self.ie_upscaling = 'bicubic'
        self.detach_estimation = detach_estimation
        
        self.ctx_ch = 3 * 64 + 3
        self.div_flow = 20.0
        self.eps = 1e-12

        self.initScaleNets_filter, self.initScaleNets_filter1, self.initScaleNets_filter2 = self.get_MonoNet5(channel, filter_size * filter_size, "filter")
        self.ctxNet = S2D_models.S2DF_3dense()
        self.rectifyNet = Resblock.MultipleBasicBlock_4(3 + 3 + 3 + 2*1 + 2*2 + 16*2 + 2*self.ctx_ch, 128)
        self.upscaleNet = Composer(scale = scale, channel = channel)
        tools.initialize_weights(self)

        self.mergeNet = AHDRNet.AHDR()
        tools.initialize_weights_xavier(self.mergeNet)

        self.flownets = PWCNet.pwc_dc_net('submodules/PWCNet/pwc_net.pth.tar') if self.training else PWCNet.pwc_dc_net()
        self.depthNet = MegaDepth.HourGlass('submodules/MegaDepth/checkpoints/test_local/best_generalization_net_G.pth') if self.training else MegaDepth.HourGlass()

        return


    def forward(self, frame_start, frame_end, sr_prev = None, low_memory = False):
        '''
        Deal with input
            - simple upscaling for improved estimations
            - padding
        '''
        assert frame_start.shape == frame_end.shape
        # self.ie_threshold*ie_scale has to fit into memory, beware of non-quadratic inputs!
        # Example: 
        #  64x64 with ie_scale 2 => 128x128: 7 steps fit into 24GB
        # 112x64 with ie_scale 2 => 224x128 => padded 256x128: 7 steps do not fit into 24GB
        ie_scale = max(1, self.ie_threshold // min(frame_start.shape[-2], frame_start.shape[-1]))
        if ie_scale > 1:
            frame_start = tools.upscale(item = frame_start, mode = self.ie_upscaling, scale_factor = ie_scale)
            frame_end = tools.upscale(item = frame_end, mode = self.ie_upscaling, scale_factor = ie_scale)

        padded_frame_start, pad = tools.pad_tensor(tensor = frame_start)
        padded_frame_end, _ = tools.pad_tensor(tensor = frame_end)

        time_offsets_rec = [0] + [ 1.0/(self.interpolated_frames-i+1) for i in range(self.interpolated_frames) ] + [1]
        time_offsets_lin = [ f/(self.interpolated_frames+1) for f in range(0, 2+self.interpolated_frames, 1) ]

        '''
        Linear: DAIN approach, all interpolations based on frame_start and frame_end
        Recurrent: inter_1 = f(frame_start, frame_end), inter_2 = f(inter_1, frame_end), ...
        '''
        if self.detach_estimation:
            with torch.no_grad():
                linear_frames = self.forward_linear(frame_start = padded_frame_start, frame_end = padded_frame_end, time_offsets = time_offsets_lin)
        else:
            linear_frames = self.forward_linear(frame_start = padded_frame_start, frame_end = padded_frame_end, time_offsets = time_offsets_lin)
        linear_frames = [padded_frame_start] + linear_frames[1:-1] + [padded_frame_end]

        outputs_lr = []
        outputs_sr = []

        def __step(frame_start, frame_end, timestep, linear_lr, last_lr, last_sr, stop_gradient):
            # Interpolate one step
            interpolated_tmp = self.forward_linear(
                frame_start = frame_start,
                frame_end = frame_end, 
                time_offsets = [timestep])

            # Merge linear and recurrent frame
            if timestep > 0 and timestep < 1:
                merged_tmp = self.forward_merge(frames_lin = [linear_lr], frames_rec = interpolated_tmp)[0]
            else:
                merged_tmp = linear_lr

            # Upscale merged frame
            if not self.upscaleNet.scale == 1:
                sr_tmp = self.forward_upscaling(lr = merged_tmp, last_lr = last_lr, last_sr = last_sr, ie_scale = ie_scale)
            else:
                sr_tmp = merged_tmp

            # Save things for output
            outputs_sr.append(sr_tmp)
            if self.training:
                unpadded_merged_tmp = tools.unpad_tensor(tensor = merged_tmp, pad = pad)
                outputs_lr.extend(tools.downscale(item = unpadded_merged_tmp, scale_factor = 1/ie_scale) if ie_scale > 1 else merged_tmp)

            if stop_gradient:
                return merged_tmp.detach(), sr_tmp.detach()
            return merged_tmp, sr_tmp

        # Initial frames for upscaling: 
        # if super-resolving sequence of more than two files, set the last output frame from the previous step as sr_prev
        lr_tmp = padded_frame_start
        sr_tmp = tools.upscale(item = padded_frame_start, scale_factor = self.upscaleNet.scale/ie_scale) if sr_prev == None else tools.pad_tensor(sr_prev)[0]
        for timestep, frame in zip(time_offsets_rec, linear_frames):
            lr_tmp, sr_tmp = __step(
                frame_start = lr_tmp,
                frame_end = padded_frame_end, 
                timestep = timestep,
                linear_lr = frame,
                last_lr = lr_tmp,
                last_sr = sr_tmp,
                stop_gradient = low_memory)

        # unpadding
        if not self.upscaleNet.scale == 1:
            outputs_sr = [ tools.unpad_tensor(tensor = t, pad = pad, scaling = self.upscaleNet.scale/ie_scale) for t in outputs_sr ]
        else:
            outputs_sr = outputs_lr

        if self.training:
            return outputs_sr, outputs_lr
        return outputs_sr
    

    '''
    Forward pass linear interpolation (original DAIN implementation)
    Input:
        - input: first and last frame, intermediate ground truth if training
    Training returns:
        - interpolated frames
        - offsets: flow estimation
        - filters: context estimation
        - losses: losses from original DAIN implementation
        - cur_ctx_output: context estimation
        - depth estimation
        - debugging frames
    Evaluation returns:
        - cur_outputs: interpolated images
        - cur_offset_output: flow estimation
        - cur_filter_output: context estimation
    '''
    def forward_linear(self, frame_start, frame_end, time_offsets):
        '''
        Estimation by Three Subpath Network
        '''
        # Flow estimation with PWCNet seems to be weak on very small frames (width/height divided by 64 at smallest point)
        cur_offset_outputs, cur_filter_output, cur_ctx_output = self.forward_estimation(
            frame_start = frame_start,
            frame_end = frame_end,
            time_offsets = time_offsets)[:3]

        '''
        Frame interpolation
        '''
        frames_output = []
        for coo_0, coo_1, time_offset in zip(cur_offset_outputs[0], cur_offset_outputs[1], time_offsets):
            frames_output_tmp = self.forward_interpolation(
                cur_ctx_output = cur_ctx_output,
                cur_offset_output = [coo_0, coo_1],
                cur_filter_output = cur_filter_output,
                frame_start = frame_start,
                frame_end = frame_end,
                time_offset = time_offset
            )
            frames_output.append(frames_output_tmp)

        return frames_output


    '''
    Perform estimation by Three Subpath Network
    Input: 
        - frame_start: frame before interpolated frame, shape (batch, 3, width, height)
        - frame_end: frame after interpolated frame, shape (batch, 3, width, height)
    Returns:
        - cur_offset_output: flow estimation, [shape (batch, 2, width, height), shape (batch, 2, width, height)]
        - cur_filter_output: context estimation, [shape (batch, width, height), shape (batch, 16, width, height)]
        - cur_ctx_output: context estimation, [shape (batch, 196, width, height), shape (batch, 196, width, height)]
        - log_depth: depth estimation, [shape (batch, 1, width, height), shape (batch, 1, width, height)]
        - depth_inv: depth estimation, [shape (batch, 1, width, height), shape (batch, 1, width, height)]
    '''
    def forward_estimation(self, frame_start, frame_end, time_offsets):        
        """
        Parameters
        ----------
        input: shape (3, batch, 3, width, height)
        -----------
        """
        s1 = torch.cuda.current_stream()
        s2 = torch.cuda.current_stream()

        time_offsets_rev = [ 1-t for t in time_offsets ]

        cur_filter_input = torch.cat((frame_start, frame_end), dim=1)
        cur_offset_input = cur_filter_input

        with torch.cuda.stream(s1):
            # MegaDepth --> Depth
            temp  = self.depthNet(torch.cat((cur_filter_input[:, :3, ...], cur_filter_input[:, 3:, ...]),dim=0))
            log_depth = [
                temp[:cur_filter_input.size(0)], 
                temp[cur_filter_input.size(0):]
                ]
            # Bootleg Resnet --> Context
            cur_ctx_output = [
                torch.cat((self.ctxNet(cur_filter_input[:, :3, ...]), log_depth[0].detach()), dim=1),
                torch.cat((self.ctxNet(cur_filter_input[:, 3:, ...]), log_depth[1].detach()), dim=1)
                ]
            temp = self.forward_singlePath(self.initScaleNets_filter, cur_filter_input, 'filter')
            # Kernel
            cur_filter_output = [
                self.forward_singlePath(self.initScaleNets_filter1, temp, name=None),
                self.forward_singlePath(self.initScaleNets_filter2, temp, name=None)
                ]

            depth_inv = [1e-6 + 1 / torch.exp(d) for d in log_depth]

        with torch.cuda.stream(s2):
            # PWCNet --> Flow
            cur_offset_outputs = [
                self.forward_flownets(
                    model = self.flownets, 
                    input = cur_offset_input, 
                    time_offsets = time_offsets),
                self.forward_flownets(
                    model = self.flownets, 
                    input = torch.cat((cur_offset_input[:, 3:, ...], cur_offset_input[:, 0:3, ...]), dim=1), 
                    time_offsets = time_offsets_rev)
                ]

        torch.cuda.synchronize() #synchronize s1 and s2

        # Flow
        cur_offset_outputs = [
            self.FlowProject(cur_offset_outputs[0], depth_inv[0]),
            self.FlowProject(cur_offset_outputs[1], depth_inv[1])
            ]

        return cur_offset_outputs, cur_filter_output, cur_ctx_output, log_depth, depth_inv


    '''
    Frame interpolation
    Input:
        - cur_ctx_output: context estimation, [shape (batch, 196, width, height), shape (batch, 196, width, height)]
        - cur_offset_output: flow estimation, [shape (batch, 2, width, height), shape (batch, 2, width, height)]
        - cur_filter_output: context estimation, [shape (batch, width, height), shape (batch, 16, width, height)]
        - frame_start: frame before interpolated frame, shape (batch, 3, width, height)
        - frame_end: frame after interpolated frame, shape (batch, 3, width, height)
    Returns:
        - [cur_output, cur_output_rectified]: interpolated frames, [shape (batch, 3, width, height), shape (batch, 3, width, height)]
    '''
    def forward_interpolation(self, cur_ctx_output, cur_offset_output, cur_filter_output, frame_start, frame_end, time_offset):
        ctx0, ctx2 = self.FilterInterpolate_ctx(
            cur_ctx_output[0],
            cur_ctx_output[1],
            cur_offset_output,
            cur_filter_output)

        cur_output, ref0, ref2 = self.FilterInterpolate(
            ref0 = frame_start, 
            ref2 = frame_end,
            offset = cur_offset_output,
            filter = cur_filter_output,
            filter_size2 = self.filter_size**2,
            time_offset = time_offset
            )

        rectify_input = torch.cat((
            cur_output,
            ref0, ref2,
            cur_offset_output[0], cur_offset_output[1],
            cur_filter_output[0], cur_filter_output[1],
            ctx0, ctx2
        ), dim =1)
        cur_output_rectified = self.rectifyNet(rectify_input) + cur_output
        
        return cur_output_rectified


    '''
    Merge linear and recurrent frames
    Input:
        - frames_lin: list of frames interpolated linearly
        - frames_rec: list of frames interpolated recurrently
    Returns:
        - frames: list of merged frames
    '''
    def forward_merge(self, frames_lin, frames_rec):
        frames = []
        for r, l in zip(frames_rec, frames_lin):
            frames.append(self.mergeNet(x1 = r, x2 = l))
        return frames


    '''
    Frame upscaling
    Based on TeCoGAN eval()
    Input:
        - lr: low resolution frame to be upscaled
        - last_lr: low resolution frame before lr, or lr if first frame of sequence
        - last_sr: high resolution frame before lr, or upsampled lr if first frame of sequence
        - flow: optical flow for frame
    Returns:
        - lr: padded low resolution frame
        - sr: upscaled frame
        - warp: warped last_sr
        - flow: optical flow
        - flow_up: upscaled optical flow
    '''
    def forward_upscaling(self, lr, last_lr, last_sr, ie_scale = False):
        a = (last_lr.size(2) - lr.size(2)) * self.upscaleNet.scale
        b = (last_lr.size(3) - lr.size(3)) * self.upscaleNet.scale
        slice_h = slice(None) if a == 0 else slice(a // 2, -a // 2)
        slice_w = slice(None) if b == 0 else slice(b // 2, -b // 2)

        # Flow computed using DAIN flow estimation, but TecoGAN warping produces more coherent edges
        if self.detach_estimation:
            with torch.no_grad():
                flow = self.forward_flownets(
                    model = self.flownets, 
                    input = torch.cat((lr, last_lr), dim=1), 
                    time_offsets = [1])[0]
        else:
            flow = self.forward_flownets(
                model = self.flownets, 
                input = torch.cat((lr, last_lr), dim=1), 
                time_offsets = [1])[0]
        
        flow_up = tools.upscale(item = flow, scale_factor = self.upscaleNet.scale//ie_scale)
        u, v = [ x.squeeze(1) for x in flow_up.split(1, dim = 1) ]
        warp = self.upscaleNet.warpper(last_sr, u, v)

        bi = None
        if ie_scale > 1:
            bi = tools.upscale(item = lr, scale_factor = self.upscaleNet.scale//ie_scale)
            lr = tools.downscale(item = lr, scale_factor = 1/ie_scale)
        
        sr = self.upscaleNet(lr = lr, sr_pre = last_sr, sr_warp = warp, preupsampled = bi)
        sr = sr[..., slice_h, slice_w]
        
        return sr


    '''
    Prepare input for discriminator
    Input:
        - sr: SR frames
        - frame0: low resolution first input frame
        - frame1: low resolution second input frame
        - temporal: warp frames
        - spatial: include unwarped frames, depth, flow and upscaled input frames
    Output:
        - stacked combination of frames, warped frames, flow and depth estimation, upscaled input frames
    '''
    def prepare_discriminator_inputs(self, sr, frame0, frame1, temporal = True, spatial = True, context = False, depth = True, flow = True):
        output = []
        if spatial:
            # channels: 3 * (interpolated_frames + 4)
            output += sr
            output += [ tools.upscale(item = frame0, scale_factor = self.upscaleNet.scale) ]
            output += [ tools.upscale(item = frame1, scale_factor = self.upscaleNet.scale) ]
        for i in range(1, len(sr) - 1):
            pre, cur, nex = sr[i-1:i+2]
            with torch.no_grad():
                # Estimations from current to previous and next frame
                if context or depth:
                    b_ctx_output, b_log_depth, b_depth_inv = self.forward_estimation(frame_start = cur, frame_end = pre, time_offsets = [1])[2:]
                    f_ctx_output, f_log_depth, f_depth_inv = self.forward_estimation(frame_start = cur, frame_end = nex, time_offsets = [1])[2:]

                if flow or temporal:
                    b_flow = self.forward_flownets(self.flownets, torch.cat((pre, cur), dim=1), [1])[0]
                    f_flow = self.forward_flownets(self.flownets, torch.cat((nex, cur), dim=1), [1])[0]
                
                if context:
                    # channels: 784 * interpolated_frames
                    output += tools.flat_list(b_ctx_output)
                    output += tools.flat_list(f_ctx_output)
                if depth:
                    # channels: 8 * interpolated_frames
                    output += tools.flat_list(b_log_depth)
                    output += tools.flat_list(b_depth_inv)
                    output += tools.flat_list(f_log_depth)
                    output += tools.flat_list(f_depth_inv)
                if flow:
                    # channels: 4 * interpolated_frames
                    output += tools.flat_list(b_flow)
                    output += tools.flat_list(f_flow)
                if temporal:
                    # channels: 9 * interpolated_frames
                    b_u, b_v = [ x.squeeze(1) for x in b_flow.split(1, dim = 1) ]
                    b_warp = self.upscaleNet.warpper(pre, b_u, b_v)
                    f_u, f_v = [ x.squeeze(1) for x in f_flow.split(1, dim = 1) ]
                    f_warp = self.upscaleNet.warpper(nex, f_u, f_v)
                    output += [ b_warp ]
                    output += [ cur ]
                    output += [ f_warp ]
        return torch.cat(output, dim = 1)


    def forward_flownets(self, model, input, time_offsets = None):
        if time_offsets == None :
            time_offsets = [0.5]
        elif type(time_offsets) == float:
            time_offsets = [time_offsets]

        temp = model(input)

        temps = [ self.div_flow * temp * time_offset for time_offset in time_offsets ]
        temps = [ tools.upscale(item = temp, scale_factor = 4) for temp in temps ]
        return temps


    '''keep this function'''
    def forward_singlePath(self, modulelist, input, name):
        stack = Stack()

        k = 0
        temp = []
        for layers in modulelist:
            # use the pop-pull logic, looks like a stack
            if k == 0:
                temp = layers(input)
            else:
                # met a pooling layer, take its input
                if isinstance(layers, nn.AvgPool2d) or isinstance(layers,nn.MaxPool2d):
                    stack.push(temp)
                temp = layers(temp)

                # met a unpooling layer, take its output
                if isinstance(layers, nn.Upsample):
                    if name == 'offset':
                        temp = torch.cat((temp,stack.pop()),dim=1)  # short cut here, but optical flow should concat instead of add
                    else:
                        temp += stack.pop()  # short cut here, but optical flow should concat instead of add
            k += 1
        return temp


    '''keep this funtion'''
    def get_MonoNet5(self, channel_in, channel_out, name):

        '''
        Generally, the MonoNet is aimed to provide a basic module for generating either offset, or filter, or occlusion.

        :param channel_in: number of channels that composed of multiple useful information like reference frame, previous coarser-scale result
        :param channel_out: number of output the offset or filter or occlusion
        :param name: to distinguish between offset, filter and occlusion, since they should use different activations in the last network layer

        :return: output the network model
        '''
        model = []

        # block1
        model += self.conv_relu(channel_in * 2, 16, (3, 3), (1, 1))
        model += self.conv_relu_maxpool(16, 32, (3, 3), (1, 1), (2, 2))  # THE OUTPUT No.5
        # block2
        model += self.conv_relu_maxpool(32, 64, (3, 3), (1, 1), (2, 2))  # THE OUTPUT No.4
        # block3
        model += self.conv_relu_maxpool(64, 128, (3, 3), (1, 1), (2, 2))  # THE OUTPUT No.3
        # block4
        model += self.conv_relu_maxpool(128, 256, (3, 3), (1, 1), (2, 2))  # THE OUTPUT No.2
        # block5
        model += self.conv_relu_maxpool(256, 512, (3, 3), (1, 1), (2, 2))

        # intermediate block5_5
        model += self.conv_relu(512, 512, (3, 3), (1, 1))

        # block 6
        model += self.conv_relu_unpool(512, 256, (3, 3), (1, 1), 2)  # THE OUTPUT No.1 UP
        # block 7
        model += self.conv_relu_unpool(256, 128, (3, 3), (1, 1), 2)  # THE OUTPUT No.2 UP
        # block 8
        model += self.conv_relu_unpool(128, 64, (3, 3), (1, 1), 2)  # THE OUTPUT No.3 UP

        # block 9
        model += self.conv_relu_unpool(64, 32, (3, 3), (1, 1), 2)  # THE OUTPUT No.4 UP

        # block 10
        model += self.conv_relu_unpool(32,  16, (3, 3), (1, 1), 2)  # THE OUTPUT No.5 UP

        # output our final purpose
        branch1 = []
        branch2 = []
        branch1 += self.conv_relu_conv(16, channel_out,  (3, 3), (1, 1))
        branch2 += self.conv_relu_conv(16, channel_out,  (3, 3), (1, 1))

        return  (nn.ModuleList(model), nn.ModuleList(branch1), nn.ModuleList(branch2))


    '''keep this function'''
    @staticmethod
    def FlowProject(inputs, depth = None):
        if depth is not None:
            outputs = [DepthFlowProjectionModule(input.requires_grad)(input,depth) for input in inputs]
        else:
            outputs = [ FlowProjectionModule(input.requires_grad)(input) for input in inputs]
        return outputs


    '''keep this function'''
    @staticmethod
    def FilterInterpolate_ctx(ctx0, ctx2, offset, filter):
        ctx0_offset = FilterInterpolationModule()(ctx0,offset[0].detach(),filter[0].detach())
        ctx2_offset = FilterInterpolationModule()(ctx2,offset[1].detach(),filter[1].detach())
        return ctx0_offset, ctx2_offset
    
    
    '''Keep this function'''    
    @staticmethod
    def FilterInterpolate(ref0, ref2, offset, filter, filter_size2, time_offset):
        ref0_offset = FilterInterpolationModule()(ref0, offset[0],filter[0])
        ref2_offset = FilterInterpolationModule()(ref2, offset[1],filter[1])
        return ref0_offset*(1.0 - time_offset) + ref2_offset*(time_offset), ref0_offset, ref2_offset


    '''keep this function'''
    @staticmethod
    def conv_relu_conv(input_filter, output_filter, kernel_size, padding):
        # we actually don't need to use so much layer in the last stages.
        layers = nn.Sequential(
            nn.Conv2d(input_filter, input_filter, kernel_size, 1, padding),
            nn.ReLU(inplace=False),
            nn.Conv2d(input_filter, output_filter, kernel_size, 1, padding)
        )
        return layers


    '''keep this fucntion'''
    @staticmethod
    def conv_relu(input_filter, output_filter, kernel_size, padding):
        layers = nn.Sequential(*[
            nn.Conv2d(input_filter, output_filter, kernel_size, 1, padding),
            nn.ReLU(inplace=False)
        ])
        return layers


    '''keep this function'''
    @staticmethod
    def conv_relu_maxpool(input_filter, output_filter, kernel_size, padding, kernel_size_pooling):
        layers = nn.Sequential(*[
            nn.Conv2d(input_filter, output_filter, kernel_size, 1, padding),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size_pooling)
        ])
        return layers


    '''keep this function'''
    @staticmethod
    def conv_relu_unpool(input_filter, output_filter, kernel_size, padding, unpooling_factor):
        layers = nn.Sequential(*[
            nn.Upsample(scale_factor=unpooling_factor, mode='bilinear', align_corners=True),
            nn.Conv2d(input_filter, output_filter, kernel_size, 1, padding),
            nn.ReLU(inplace=False)
        ])
        return layers