# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os, json, random, zipfile
import re
from typing import List, Optional

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
    num = min(int(len(idlist)*0.7), 100)
    rowlist = idlist[:num]
    return rowlist

def generate_images(
    G, seeds, outdir, truncation_psi=1.0, noise_mode='const', class_idx=None
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate curated MetFaces images without truncation (Fig.10 left)
    python generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate uncurated MetFaces images with truncation (Fig.12 upper left)
    python generate.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate class conditional CIFAR-10 images (Fig.17 left, Car)
    python generate.py --outdir=out --seeds=0-35 --class=1 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl

    \b
    # Render an image from projected W
    python generate.py --outdir=out --projected_w=projected_w.npz \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
    """
    # seeds = range(100)####设定seed？

    os.makedirs(outdir, exist_ok=True)
    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print('warn: --class=lbl ignored when running on an unconditional network')

    # Generate images.
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')

@click.command()
@click.option('--network', 'networkpath', help='Network pickle filename', required=True)
@click.option('--clsnum', 'classes', type=int, required=True)
@click.option('--jsonpath', 'jsonpath', type=str, required=True)
@click.option('--outdir', 'outpdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
def main(networkpath: str, classes: int, outpdir: str, jsonpath: str):
    ##python autogen.py --network=runs/car/00019-dem8tr-cond-mirror-paper256-kimg10000-batch128-bgcfnc-resumecustom/network-006144.pkl --clsnum=8 --jsonpath=../datasets/dem8tr.zip --outdir=../datasets/mix4/train/
    print('Loading networks from "%s"...' % networkpath)
    with dnnlib.util.open_url(networkpath) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore
    # cls2dict = {1: 0, 3: 1, 2: 2, 0: 3} if classes==4 else {0:4, 1:5, 2:1, 3:7, 4:2, 5:0, 6:6, 7:3}
    for i in range(classes):
        outdir = outpdir + "%02d" % i  ###':02>d'.format(i)
        rowlist = get_rowcol(jsonpath, i)
        print('check rowlist ', rowlist)
        generate_images(G, rowlist, outdir, class_idx=i)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
