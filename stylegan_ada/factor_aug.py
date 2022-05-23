import os, json, random, zipfile
import torch, pickle
from torchvision import utils
import dnnlib
import PIL.Image
import numpy as np
import click
device = torch.device('cuda')
import legacy
# from model import Generator
# def get_factor(ckptpath):
#     with dnnlib.util.open_url(ckptpath) as f:
#         ckpt = legacy.load_network_pkl(f)##['G_ema'].to(device)  # type: ignore
#     print('check ckpt', ckpt['G'], type(ckpt['G']))
#     modulate = {
#         k: v
#         for k, v in ckpt["G_ema"].items()
#         if "modulation" in k and "to_rgbs" not in k and "weight" in k
#     }
#
#     weight_mat = []
#     for k, v in modulate.items():
#         weight_mat.append(v)
#
#     W = torch.cat(weight_mat, 0)
#     eigvec = torch.svd(W).V.to("cpu")
#
#     # torch.save({"ckpt": ckpt, "eigvec": eigvec}, out)
#     return eigvec
'''https://github.com/pbizimis/stylegan2-ada-pytorch的将stylegan2ada model提取重要特征的实现'''
def get_factor2(ckptpath):
    with dnnlib.util.open_url(ckptpath) as f:
        G = pickle.load(f)['G_ema']#.to(device)  # type: ignore

    modulate = {
        k[0]: k[1]
        for k in G.named_parameters()
        if "affine" in k[0] and "torgb" not in k[0] and "weight" in k[0] or (
                    "torgb" in k[0] and "b4" in k[0] and "weight" in k[0] and "affine" in k[0])
    }

    weight_mat = []
    for k, v in modulate.items():
        weight_mat.append(v)

    W = torch.cat(weight_mat, 0)
    eigvec = torch.svd(W).V.to("cpu")####奇异分解，排序
    slist = torch.diag(torch.diag(torch.svd(W).S.to("cpu")))###对角线元素
    print('check diag list ', slist, 'eigvec shape', eigvec.shape)
    diagsum = torch.sum(slist)
    svdval = 0
    for i in range(len(slist)):
        # tdiags = torch.sum(slist[:i])
        # if tdiags > 0.8*diagsum:
        #     svdval = i
        #     break
        if slist[i]/diagsum < 0.005:
            svdval = i
            break
    print('get factors, with ', svdval, ' major factors', ' 16 factors weight ', slist[:16]/diagsum)
    # torch.save({"ckpt": args.ckpt, "eigvec": eigvec}, args.out)
    return eigvec

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

def zs_to_ws(G,label,truncation_psi,zs):
    ws = []
    for z_idx, z in enumerate(zs):
        # z = torch.from_numpy(z).to(device)
        w = G.mapping(z, label, truncation_psi=truncation_psi, truncation_cutoff=8)
        ws.append(w)
    return ws

def generate_images(G, eigvec, seeds, outdir, label, truncation_psi):
    os.makedirs(outdir, exist_ok=True)
    for l in seeds:
        # print(f"Generate images for seed ", l)
        z = torch.from_numpy(np.random.RandomState(l).randn(1, G.z_dim)).to(device)
        index_list_of_eigenvalues = range(8)###单纯方便后面使用

        for j in index_list_of_eigenvalues:
            current_eigvec = eigvec[:, j].unsqueeze(0)
            for degree in [-10, 10]:
                direction = degree * current_eigvec
                direction.to(device)
                # image_group = generate_images(G, z, label, truncation_psi, noise_mode, direction)
                img = G(z+direction, label, truncation_psi=truncation_psi, noise_mode='const')
                file_name = f"{outdir}/seed{l}-{j}_{degree}.png"
                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(file_name)

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
    eigvec = get_factor2(networkpath).to(device)
    # cls2dict = {1: 0, 3: 1, 2: 2, 0: 3} if classes==4 else {0:4, 1:5, 2:1, 3:7, 4:2, 5:0, 6:6, 7:3}
    if not os.path.exists(outpdir):
        os.mkdir(outpdir)
    for i in range(classes):
        outdir = outpdir + "%02d" % i  ###':02>d'.format(i)
        rowlist = get_rowcol(jsonpath, i)
        print('check rowlist ', rowlist)
        label = torch.zeros([1, G.c_dim], device=device)  # assume no class label
        # Labels.
        label[:, i] = 1

        generate_images(G, eigvec, rowlist, outdir, label, truncation_psi=0.7)

if __name__ == "__main__":
    main()
