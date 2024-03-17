# demo inspired by https://huggingface.co/spaces/lambdalabs/image-mixer-demo
import argparse
import copy
import cv2
#import gradio as gr
import torch
from functools import partial
from itertools import chain
from torch import autocast
import sys
import cv2
sys.path.append('/mnt/ssd1/chencheng/threestudio/threestudio/models/T2I_ada')

from basicsr.utils import tensor2img
from ldm.inference_base import DEFAULT_NEGATIVE_PROMPT, diffusion_inference, get_adapters, get_sd_models, forward_unet
from ldm.modules.extra_condition import api
from ldm.modules.extra_condition.api import ExtraCondition, get_cond_model
from ldm.modules.encoders.adapter import CoAdapterFuser
import pdb as p

torch.set_grad_enabled(False)

supported_cond = ['style', 'sketch', 'color', 'depth', 'canny']

supported_cond = ['depth', 'style']


# config
parser = argparse.ArgumentParser()
parser.add_argument(
    '--sd_ckpt',
    type=str,
    default='/mnt/ssd1/chencheng/threestudio/threestudio/models/T2I_ada/models/v1-5-pruned-emaonly.ckpt',
    help='path to checkpoint of stable diffusion model, both .ckpt and .safetensor are supported',
)
parser.add_argument(
    '--vae_ckpt',
    type=str,
    default=None,
    help='vae checkpoint, anime SD models usually have seperate vae ckpt that need to be loaded',
)


#这一行是自己加的，用来防止使用threestudio时出现未定义参数
# 解析命令行参数，但捕获未预定义的参数
global_opt,unknown_args = parser.parse_known_args()
#global_opt = parser.parse_args()

global_opt.config = '/mnt/ssd1/chencheng/threestudio/threestudio/models/T2I_ada/configs/stable-diffusion/sd-v1-inference.yaml'
for cond_name in supported_cond:
    setattr(global_opt, f'{cond_name}_adapter_ckpt', f'/mnt/ssd1/chencheng/threestudio/threestudio/models/T2I_ada/models/coadapter-{cond_name}-sd15v1.pth')
global_opt.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
global_opt.max_resolution = 512 * 512
global_opt.sampler = 'ddim'
global_opt.cond_weight = 1.0
global_opt.C = 4
global_opt.f = 8
#TODO: expose style_cond_tau to users
global_opt.style_cond_tau = 1.0

# stable-diffusion model
sd_model, sampler = get_sd_models(global_opt)
# adapters and models to processing condition inputs
adapters = {}
cond_models = {}

torch.cuda.empty_cache()

# fuser is indispensable
coadapter_fuser = CoAdapterFuser(unet_channels=[320, 640, 1280, 1280], width=768, num_head=8, n_layes=3)
coadapter_fuser.load_state_dict(torch.load(f'/mnt/ssd1/chencheng/threestudio/threestudio/models/T2I_ada/models/coadapter-fuser-sd15v1.pth'))
coadapter_fuser = coadapter_fuser.to(global_opt.device)


from PIL import Image
import numpy as np

#这里我在调用的时候只需要传入prompt, style image 和 depth image即可
opt = copy.deepcopy(global_opt)
opt.neg_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
opt.scale = 7.5
#opt.n_samples = int(n_samples.value)
#opt.seed = int(seed.value)
opt.steps = 50
opt.resize_short_edge = 512
opt.cond_tau = 1.0


def read_img(image_path):
    #image_path = "your_image_path.jpg"
    image = Image.open(image_path)
    image = image.convert("RGB")
    image_array = np.array(image)#.astype(np.float32)
    return image_array


def t2i_coada(prompt, depth_img, style_image,time_step):
    opt.prompt = prompt

    opt.b1 = "Image"  # 设置 btn1 的默认值
    #opt.depth_im = read_img(depth_img)  # 设置 im1  "/mnt/ssd1/chencheng/pnp-diffusers/data/yellow_duck_toy"
    opt.depth_im = read_img("/mnt/ssd1/chencheng/pnp-diffusers/data/yellow_duck_toy/back.png") 
    opt.depth_cond_weight = 1.0  # 设置 cond_weight 的默认值

    opt.b2 = "Image"  # 设置 btn1 的默认值
    #opt.style_im = read_img(style_image)  # 设置 im1 
    opt.style_im = read_img("/mnt/ssd1/chencheng/pnp-diffusers/data/yellow_duck_toy/back.png")
    opt.style_cond_weight = 1.0  # 设置 cond_weight 的默认值

    opt.n_samples=10

    with torch.inference_mode(), \
            sd_model.ema_scope(), \
            autocast('cuda'):
    #inps = [[opt.b1, opt.im1, opt.cond_weight1],[opt.b2, opt.im2, opt.cond_weight2]]
        inps = [[opt.b1, opt.b2],[opt.depth_im,opt.style_im],[opt.depth_cond_weight, opt.style_cond_weight]]
        #p.set_trace()

        conds = []
        activated_conds = []
        prev_size = None
        for idx, (b, im1, cond_weight) in enumerate(zip(*inps)):
            #p.set_trace()
            cond_name = supported_cond[idx]
            #p.set_trace()
            if b == 'Nothing':
                if cond_name in adapters:
                    adapters[cond_name]['model'] = adapters[cond_name]['model'].cpu()
            else:
                activated_conds.append(cond_name)
                if cond_name in adapters:
                    adapters[cond_name]['model'] = adapters[cond_name]['model'].to(opt.device)
                else:
                    adapters[cond_name] = get_adapters(opt, getattr(ExtraCondition, cond_name))
                adapters[cond_name]['cond_weight'] = cond_weight

                process_cond_module = getattr(api, f'get_cond_{cond_name}')

                if b == 'Image':
                    if cond_name not in cond_models:
                        cond_models[cond_name] = get_cond_model(opt, getattr(ExtraCondition, cond_name))
                    if prev_size is not None:
                        image = cv2.resize(im1, prev_size, interpolation=cv2.INTER_LANCZOS4)
                    else:
                        image = im1
                    conds.append(process_cond_module(opt, image, 'image', cond_models[cond_name]))
                    if idx != 0 and prev_size is None:  # skip style since we only care spatial cond size
                        h, w = image.shape[:2]
                        prev_size = (w, h)
                    print('cond_name: ',cond_name)

        features = dict()
        for idx, cond_name in enumerate(activated_conds):
            cur_feats = adapters[cond_name]['model'](conds[idx])
            if isinstance(cur_feats, list):
                for i in range(len(cur_feats)):
                    cur_feats[i] *= adapters[cond_name]['cond_weight']
            else:
                cur_feats *= adapters[cond_name]['cond_weight']
            features[cond_name] = cur_feats

        adapter_features, append_to_context = coadapter_fuser(features)

        #for i in range(opt.n_samples):
            #p.set_trace() #device(type='cuda', index=0)
        result = forward_unet(opt, sd_model, sampler, adapter_features, append_to_context,time_step=time_step)
            #p.set_trace()
            #这里用来可视化
            #ims = tensor2img(result, rgb2bgr=True)
            #cv2.imwrite(f'/mnt/ssd1/chencheng/threestudio/outputs/t2i_ada/output_{i}.jpg', ims)
        # Clear GPU memory cache so less likely to OOM
        torch.cuda.empty_cache()

t2i_coada('a yellow rubber duck, back view',1,1)