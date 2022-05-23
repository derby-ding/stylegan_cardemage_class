import argparse, os, random
import torch, pickle
from torchvision import utils
import dnnlib
import numpy as np
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
def get_factor2(G):
    # with dnnlib.util.open_url(ckptpath) as f:
    #     G = pickle.load(f)['G_ema']#.to(device)  # type: ignore

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
    eigvec = torch.svd(W).V.to("cpu")
    slist = torch.diag(torch.diag(torch.svd(W).S.to("cpu")))###对角线元素
    # print('check diag ', slist)
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

def zs_to_ws(G,label,truncation_psi,zs):
    ws = []
    for z_idx, z in enumerate(zs):
        # z = torch.from_numpy(z).to(device)
        w = G.mapping(z, label, truncation_psi=truncation_psi, truncation_cutoff=8)
        ws.append(w)
    return ws

def generate_images(G, z, label, truncation_psi, noise_mode, direction):
    # if(args.space == 'w'):
    #     ws = zs_to_ws(G, label, truncation_psi, [z, z + direction, z - direction])
    #     img1 = G.synthesis(ws[0], noise_mode=noise_mode, force_fp32=True)
    #     img2 = G.synthesis(ws[1], noise_mode=noise_mode, force_fp32=True)
    #     img3 = G.synthesis(ws[2], noise_mode=noise_mode, force_fp32=True)
    # else:
    # print('check type generate ', type(z), type(label), type(truncation_psi), type(noise_mode))
    img1 = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
    img2 = G(z + direction, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
    img3 = G(z - direction, label, truncation_psi=truncation_psi, noise_mode=noise_mode)

    return torch.cat([img3, img1, img2], 0)

if __name__ == "__main__":
    torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser(description="Apply closed form factorization")
    # parser.add_argument("-i", "--index", type=int, default=0, help="index of eigenvector")
    parser.add_argument("-d", "--degree", type=float, default=10, help="scalar factors for moving latent vectors along eigenvector",)
    parser.add_argument("--seeds", nargs='+', type=int, default=[0, 10, 20, 30], help="seeds for latent code")
    parser.add_argument("--channel_multiplier", type=int, default=2, help='channel multiplier factor. config-f = 2, else = 1')
    parser.add_argument("--ckpt", type=str, required=True, help="stylegan2 checkpoints")
    parser.add_argument("--size", type=int, default=256, help="output image size of the generator")
    parser.add_argument("-n", "--n_sample", type=int, default=7, help="number of samples created")
    parser.add_argument("--class_idx", type=int, default=0, help="specify generated class ")
    parser.add_argument("--truncation", type=float, default=0.7, help="truncation factor")
    parser.add_argument("--output", type=str, default='out', help="outputdir")
    args = parser.parse_args()

    with dnnlib.util.open_url(args.ckpt) as f:
        G = legacy.load_network_pkl(f)['G_ema']  # type: ignore
    # eigvec = torch.load(args.factor)["eigvec"].to(args.device)
    eigvec = get_factor2(G).to(device)
    G = G.to(device)
    # eigvec = torch.load(args.factor)["eigvec"].to(device)
    # index = args.index
    seeds = args.seeds####设定种子
    label = torch.zeros([1, G.c_dim], device=device)  # assume no class label
    # Labels.
    if G.c_dim != 0:
        label[:, args.class_idx] = 1
    else:
        if args.class_idx is not None:
            print('warn: --class=lbl ignored when running on an unconditional network')

    noise_mode = "const"  # default
    truncation_psi = args.truncation
    # truncation_psi.to(device)
    index_list_of_eigenvalues = []
    latents = seeds
    # file_name = ""

    for l in latents:
        print(f"Generate images for seed ", l)
        image_grid_eigvec = []
        z = torch.from_numpy(np.random.RandomState(l).randn(1, G.z_dim)).to(device)
        # if index == -1:  # use all eigenvalues
        #     index_list_of_eigenvalues = [*range(len(eigvec))]
        #     file_name = f"seed-{l}_index-all_degree-{args.degree}.png"
        # else:  # use certain indexes as eigenvalues
        index_list_of_eigenvalues = range(16)###单纯方便后面使用
        # str_index_list = '-'.join(str(x) for x in index)
        file_name = f"seed-{l}_cls-{args.class_idx}-degree-{args.degree}.png"

        for j in index_list_of_eigenvalues:
            current_eigvec = eigvec[:, j].unsqueeze(0)
            direction = args.degree * current_eigvec
            direction.to(device)
            image_group = generate_images(G, z, label, truncation_psi, noise_mode, direction)
            image_grid_eigvec.append(image_group)

        # print("Saving image ", os.path.join(args.output, file_name))
        grid = utils.save_image(torch.cat(image_grid_eigvec, 0), os.path.join(args.output, file_name), nrow=3, normalize=True, range=(-1, 1))  # change range to value_range for latest torchvision

    # ckpt = torch.load(args.ckpt)
    # g = Generator(args.size, 512, 8, channel_multiplier=args.channel_multiplier).to(args.device)
    # g.load_state_dict(ckpt["g_ema"], strict=False)

    # trunc = g.mean_latent(4096)
    # latent = torch.randn(args.n_sample, 512, device=args.device)###shape n_sample*512，随机初始特征
    # latent = g.get_latent(latent)##
    #
    # for i in range(50):
    #     args.index = i
    #     direction = args.degree * eigvec[:, args.index].unsqueeze(0)
    #     img, _ = g(
    #         [latent],
    #         truncation=args.truncation,
    #         truncation_latent=trunc,
    #         input_is_latent=True,
    #     )
    #     img1, _ = g(
    #         [latent + direction],
    #         truncation=args.truncation,
    #         truncation_latent=trunc,
    #         input_is_latent=True,
    #     )
    #     img2, _ = g(
    #         [latent - direction],
    #         truncation=args.truncation,
    #         truncation_latent=trunc,
    #         input_is_latent=True,
    #     )
    #
    #     grid = utils.save_image(torch.cat([img1, img, img2], 0), f"sample/{args.out_prefix}_index-{args.index}_degree-{args.degree}.png", normalize=True, range=(-1, 1), nrow=args.n_sample,)
