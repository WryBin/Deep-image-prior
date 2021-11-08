from __future__ import print_function

import os
import argparse

import torch

from skimage.measure import compare_psnr

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True  # Cause cuDNN to benchmark multiple convolution algorithms and select the fastest.
dtype = torch.cuda.FloatTensor

def closure():
    global i, net_input

    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)

    out_HR = net(net_input)
    out_LR = downsampler(out_HR)

    total_loss = mse(out_LR, img_LR_Var)

    # TODO
    if tv_weight > 0:
        total_loss += tv_weight * tv_loss(out_HR)

    total_loss.backward()

    # Log
    psnr_LR = compare_psnr(imgs['LR_np'], torch_to_np(out_LR))
    psnr_HR = compare_psnr(imgs['HR_np'], torch_to_np(out_HR))
    # TODO
    print('Iteration %05d   PSNR_LR %.3f    PSNR_HR %.3f' % (i, psnr_LR, psnr_HR), 'r', end='')

    # History
    psnr_history.append([psnr_LR, psnr_HR])

    if PLOT and i % 100 == 0:
        out_HR_np = torch_to_np(out_HR)
        plot_image_grid([imgs['HR_np'], imgs['bicubic_np'], np.clip(out_HR_np, 0, 1)], factor=13, nrow=3)

    i += 1

    return total_loss



# Load image and baselines
imsize = -1  # no resizing
factor = 4  # downscaling factor
enforse_div32 = 'CROP'
path_to_image = 'data/sr/zebra_GT.png'

PLOT = TRUE

imgs = load_LR_HR_imgs_sr(path_to_image, imsize, factor, enforse_div32)
imgs['bicubic_np'], imgs['sharp_np'], imgs['nearest_np'] = get_baselines(imgs['LR_pil'], imgs['HR_pil'])

if PLOT:
    plot_image_grid([imgs['HR_np'], imgs['bicubic_np'], imgs['sharp_np'], imgs['nearest_np']], 4, 12)
    print("PSNR bicubic: %.4f    PSNR nearest:%.4f" % (compare_psnr(imgs['HR_np'], imgs['bicubic_np']), compare_psnr(imgs['HR_np'], imgs['nearest_np'])))

# Set up parameters and net
input_depth = 32
INPUT = 'noise'

pad = 'reflection'

OPT_OVER = 'net'
KERNEL_TYPE = 'lanczos2'

LR = 0.01
tv_weight = 0.0

OPTIMIZER = 'adam'

if factor == 4:
    num_iter = 2000
    reg_noise_std = 0.03
elif factor == 8:
    num_iter = 4000
    reg_noise_std = 0.05
else:
    assert False, 'We did not experiment with other factors'

# TODO:Try to change the sequence
net_input = get_noise(input_depth, INPUT, (imgs['HR_pil'].size[1], imgs['HR_pil'].size[0])).type(dtype).detach()

NET_TYPE = 'skip'  # UNet, ResNet
net = get_net(input_depth, 'skip', pad,
              skip_n33d=128,
              skip_n33u=128,
              skip_n11=4,
              num_scales=5,
              upsample_mode='bilinear').type(dtype)

# losses
mse = torch.nn.MSELoss().type(dtype)

img_LR_var = np_to_torch(imgs['LR_np']).type(dtype)

downsampler = Downsampler(n_planes=3, factor=factor, kernel_type=KERNEL_TYPE, phase=0.5, preserve_size=True).type(dtype)

# Define closure and optimize

psnr_history = []
net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()

i = 0
p = get_params(OPT_OVER, net, net_input)

optimize(OPTIMIZER, p, closure, LR, num_iter)

out_HR_np  = np.clip(torch_to_np(net(net_input)), 0, 1)
result_deep_prior = put_in_center(out_HR_np, imgs['orig_np'].shape[1:])

# For the paper we actually took '_bicubic.png' files from LapSRN viewer and used 'result_deep_prior' as our result

plot_image_grid([imgs['HR_np'],
                 imgs['bicubic_np'],
                 out_HR_np], factor=4, nrow=1)



