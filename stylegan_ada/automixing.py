# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate style mixing image matrix using pretrained network pickle."""
import os
import re
from typing import List
import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import legacy
device = torch.device('cuda')
#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]
import json, random, zipfile
#----------------------------------------------------------------------------
def get_rowcol(jsonpath, i):
    with zipfile.ZipFile(jsonpath).open('dataset.json', 'r') as f:
        filenames = json.load(f)['labels']
    idlist = []
    for filen in filenames:
        if filen[1]==i:
            fileid = int(os.path.split(filen[0])[-1].split('.')[0])
            idlist.append(fileid)
    random.shuffle(idlist)
    num = min(int(len(idlist)*0.7), 50)###最多50个
    rowlist = idlist[:num]

    collist = idlist[len(idlist)-num:]
    return rowlist, collist

def generate_style_mix(G, row_seeds, col_seeds, outdir, col_styles=[0,1,2,3], truncation_psi=1.0, noise_mode='const', classid=None):
    """Generate images using pretrained network pickle.
    Examples:
    \b
    python style_mixing.py --outdir=out --rows=85,100,75,458,1500 --cols=55,821,1789,293 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
    """
    os.makedirs(outdir, exist_ok=True)
    print('Generating W vectors...')
    # col_seeds = list(range(1, 91, 3))
    # row_seeds = list(range(0, 31, 9))
    all_seeds = list(set(row_seeds + col_seeds))
    labels = torch.zeros([len(all_seeds), G.c_dim], device=device)
    if classid is not None:
        labels[:, classid] = 1
    # print('check all_seeds', all_seeds, 'generate dim', G.z_dim)
    all_z = np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in all_seeds])
    # print('check all_z', all_z.shape)
    # all_w = G.mapping(torch.from_numpy(all_z).to(device), None)
    all_w = G.mapping(torch.from_numpy(all_z).to(device), labels)###torch.cat([label] * len(all_seeds), dim=0))
    w_avg = G.mapping.w_avg
    all_w = w_avg + (all_w - w_avg) * truncation_psi
    w_dict = {seed: w for seed, w in zip(all_seeds, list(all_w))}

    print('Generating images...')####
    all_images = G.synthesis(all_w, noise_mode=noise_mode)
    all_images = (all_images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
    image_dict = {(seed, seed): image for seed, image in zip(all_seeds, list(all_images))}

    print('Generating style-mixed images...')
    # for row_seed in row_seeds:
    #     for col_seed in col_seeds:
    for row_seed, col_seed in zip(row_seeds, col_seeds):
        w = w_dict[row_seed].clone()
        w[col_styles] = w_dict[col_seed][col_styles]
        image = G.synthesis(w[np.newaxis], noise_mode=noise_mode)
        image = (image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        image_dict[(row_seed, col_seed)] = image[0].cpu().numpy()

    for col_seed, row_seed in zip(row_seeds, col_seeds):###换位再生成
        w = w_dict[row_seed].clone()
        w[col_styles] = w_dict[col_seed][col_styles]
        image = G.synthesis(w[np.newaxis], noise_mode=noise_mode)
        image = (image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        image_dict[(row_seed, col_seed)] = image[0].cpu().numpy()

    print('Saving images...')
    os.makedirs(outdir, exist_ok=True)
    for (row_seed, col_seed), image in image_dict.items():
        # if not row_seed == col_seed:####跳过重复
        PIL.Image.fromarray(image, 'RGB').save(f'{outdir}/{row_seed}-{col_seed}.png')

    # print('Saving image grid...')
    # W = G.img_resolution
    # H = G.img_resolution
    # # grid_seeds = list(set(row_seeds + col_seeds))
    # # image_dict = {(seed, seed): image for seed, image in zip(grid_seeds, list(all_images))}
    # canvas = PIL.Image.new('RGB', (W * (4 + 1), H * 2), 'black')
    # # for row_idx, row_seed in enumerate([0] + row_seeds):
    # #     for col_idx, col_seed in enumerate([0] + col_seeds):
    # rid = 1
    # flag = -1
    #
    # for col_seed, row_seed in zip(row_seeds, col_seeds):
    #     if flag < 0:
    #         flag = col_seed
    #         canvas.paste(PIL.Image.fromarray(image_dict[(row_seed, row_seed)], 'RGB'), (W * 0, H * 1))
    #         # if row_idx == 0 and col_idx == 0:
    #         #     continue
    #         # key = (row_seed, col_seed)
    #         # if row_idx == 0:
    #         #     key = (col_seed, col_seed)
    #         # if col_idx == 0:
    #         #     key = (row_seed, row_seed)
    #     if col_seed == flag:
    #         key = (col_seed, row_seed)
    #         canvas.paste(PIL.Image.fromarray(image_dict[(col_seed, col_seed)], 'RGB'), (W * rid, H * 0))
    #         canvas.paste(PIL.Image.fromarray(image_dict[key], 'RGB'), (W * rid, H * 1))
    #         rid += 1
    # canvas.save(f'{outdir}/grid.png')

import os, cv2
import click
@click.command()
@click.option('--network', 'networkpath', type=str, required=True)
@click.option('--clsnum', 'classes', type=int, required=True)
@click.option('--jsonpath', 'jsonpath', type=str, required=True)
@click.option('--outdir', 'outpdir', type=str, required=True)
#----------------------------------------------------------------------------
def main(networkpath: str, classes: int, outpdir: str, jsonpath:str):
    '''python automixing.py --network=runs/car/00019-dem8tr-cond-mirror-paper256-kimg10000-batch128-bgcfnc-resumecustom/network-002048.pkl --clsnum=8 --jsonpath=../datasets/dem8tr.zip --outdir=../datasets/mix8/train/
'''
    # classes = 4
    # networkpath = 'runs/car/00029-cls8gan128-cond-mirror-cifar-kimg10000-batch128-bgcfnc/network-010000.pkl'
    # jsonpath = '../datasets/cls8gan128.zip'
    # outpdir = '../datasets/mix8r128/train/'  ####cp -r src outpdir
    print('Loading networks from "%s"...' % networkpath)
    with dnnlib.util.open_url(networkpath) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore
    # cls2dict = {1: 0, 3: 1, 2: 2, 0: 3} if classes==4 else {0:4, 1:5, 2:1, 3:7, 4:2, 5:0, 6:6, 7:3}####类型和文件夹不匹配
    for i in range(classes):
        outdir = outpdir + "%02d" % i  ###':02>d'.format(i)
        rowlist, collist = get_rowcol(jsonpath, i)
        print('check rowlist ', rowlist)
        generate_style_mix(G, rowlist, collist, outdir, classid=i)  # pylint: disable=no-value-for-parameter
if __name__ == "__main__":
    ####cp -r datasets/demcls8/train/ datasets/mix8r28/train
#####cls 8
    # classes = 8
    # networkpath = 'runs/car/00029-cls8gan128-cond-mirror-cifar-kimg10000-batch128-bgcfnc/network-010000.pkl'
    # jsonpath = '../datasets/cls8gan128.zip'
    # outpdir = '../datasets/mix8r28/train/'  ####cp -r src outpdir
    # print('Loading networks from "%s"...' % networkpath)
####cls 4
    main()

#----------------------------------------------------------------------------
