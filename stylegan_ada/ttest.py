import os, pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 1, sharey=False, figsize=(6, 4))
# x = [0, 2048, 4096, 6144, 8192, 10240, 12288, 15000]##8
# yc = [325.0, 180.1, 132.9, 105.1, 93.7, 80.6, 83.1, 74.3]
# yf = [325.0, 206.3, 130.5, 99.9, 83.1, 74.8, 71.4, 71.0]
# yb = [320.1, 183.1, 129.8, 90.3, 78.0, 75.4, 76.3, 70.8]
# yg = [320.2, 183.0, 132.4, 94.8, 77.9, 74.2, 60.4, 61.8]
# yn = [325.1, 199.2, 131.4, 112.5, 98.3, 90.5, 85.9, 82.5]
# yct = [325.0, 202.6, 101.7, 89.3, 77.3, 74.8, 81.4, 73.1]
# au6 = [325.2, 200.9, 153.5, 122.8, 86.5, 78.0, 78.4, 88.7]
# ada = [325.2, 99.1, 57.4, 49.5, 45.8, 44.6, 43.9, 45.1]
#
# # df = pd.DataFrame({'x':x, 'yc':yc, 'yb':yb, 'yg':yg, 'yf':yf, 'yn':yn, 'yct':yct, 'aug':au6, 'ada':ada})
# sns.lineplot(x=x, y=yf, ax=axs, marker='o', color='b', legend='brief', label='fliter')
# sns.lineplot(x=x, y=yg, ax=axs, marker='*', color='r', legend='auto', label='geom')
# sns.lineplot(x=x, y=yn, ax=axs, marker='*', color='g', legend='auto', label='noise')
# sns.lineplot(x=x, y=yct, ax=axs, marker='o', color='r', legend='auto', label='crop')
# sns.lineplot(x=x, y=yc, ax=axs, marker='o', color='g', legend='auto', label='color')
# sns.lineplot(x=x, y=yb, ax=axs, marker='*', color='b', legend='auto', label='blit')
# sns.lineplot(x=x, y=au6, ax=axs, marker='o', color='y', legend='auto', label='bgcnfc')
# sns.lineplot(x=x, y=ada, ax=axs, marker='*', color='y', legend='auto', label='ada')
# axs.set_xlabel('kimgs for Training')
# axs.set_ylabel('Fid score')
# plt.savefig('dataaugfid.png')

df = pd.read_excel('F:/carinsure_cv/pytorch-image-models/result8128.xls')[::10]
x = range(0, 210, 10)
df['x'] = x
sns.lineplot(data=df, x='x', y='base', ax=axs, marker='*', color='y', label='original train set')
sns.lineplot(data=df, x='x', y='aug', ax=axs, marker='*', color='g', label='+random generation mode')
sns.lineplot(data=df, x='x', y='mix', ax=axs, marker='*', color='b', label='+mixing generation mode')
sns.lineplot(data=df, x='x', y='aug+mix', ax=axs, marker='*', color='r', label='+mixing+random mode')
sns.lineplot(data=df, x='x', y='disentangle', ax=axs, marker='o', color='b', label='+disentange mode')
axs.set_xlabel('Model Training epochs')
axs.set_ylabel('Classification Accuracy')
# axs[0].set_ylim([70, 330])####需根据数据动态修改
# # axt = axs[0].twinx()
# sns.lineplot(x=x, y=yb, ax=axs[0], marker='*', color='g')
# axs[0].tick_params(axis='x', labelrotation=90)
# axs[0].tick_params(axis='x', labelrotation=90).set_title('IsFraud')
# axs[1].tick_params(axis='x', labelrotation=90).set_title('NotFraud')
plt.savefig('cls8acc0.png')
exit()
# run_dir = 'out/test'
# snapshot_data = {'data': [np.random.randn(80, 100)]}
# # snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl')###cover
# snapshot_pkl = os.path.join(run_dir, f'network0.pkl')
# for i in range(10):
#     with open(snapshot_pkl, 'wb') as f:###覆盖
#         pickle.dump(snapshot_data, f)
import json, zipfile, torch
outpdir = '../datasets/mix8r28/train/'
fname = 'dataset.json'
labels = torch.zeros([2, 5])
labels[:, 3] = 1
print('check labels', labels)
exit()
# zipfile.ZipFile('../datasets/dem8tr.zip')
with zipfile.ZipFile('../datasets/cls8gan128.zip').open(fname, 'r') as f:
    labels = json.load(f)['labels']
print('check load_raw labels************', fname, labels)
exit()
import dnnlib
args = dnnlib.EasyDict()
args.training_set_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path='../datasets/cls8gan128.zip', use_labels=True, max_size=None, xflip=False)
args.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=3, prefetch_factor=2)

training_set = dnnlib.util.construct_class_by_name(**args.training_set_kwargs) # subclass of training.dataset.Dataset
args.training_set_kwargs.resolution = training_set.resolution # be explicit about resolution
args.training_set_kwargs.use_labels = training_set.has_labels # be explicit about labels
print('check args', args)
exit()


import legacy
import torch
import PIL
from PIL import Image
device = torch.device('cuda')
# Code to load the StyleGAN2 Model
def load_model(modelpath):
    with dnnlib.util.open_url(modelpath) as f:
        model = legacy.load_network_pkl(f)
    _G, _D, Gs = model['G'], model['D'], model['G_ema']

    # noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
    #
    # Gs_kwargs = dnnlib.EasyDict()
    # Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    # Gs_kwargs.randomize_noise = False

    return Gs.to(device)


# Generate images given a random seed (Integer)
def generate_image_random(G, seed, outdir, trun=1, noise_mode='const'):
    # rnd = np.random.RandomState(rand_seed)
    # z = rnd.randn(1, *Gs.input_shape[1:])
    # tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars})
    # images = Gs.run(z, None, **Gs_kwargs)
    label = torch.zeros([1, G.c_dim], device=device)
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
    img = G(z, label, truncation_psi=trun, noise_mode=noise_mode)
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')

    return PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB'), z

# Generate images given a latent code ( vector of size [1, 512] )
def generate_image_from_z(G, z):
    # images = Gs.run(z, None, **Gs_kwargs)
    label = torch.zeros([1, G.c_dim], device=device)
    # z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
    img = G(z, label)
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return img

def linear_interpolate(code1, code2, alpha):
    return code1 * alpha + code2 * (1 - alpha)


def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def make_latent_interp_animation(G, code1, code2, img1, img2, num_interps, outdir):
    step_size = 1.0 / num_interps
    all_imgs = []
    amounts = np.arange(0, 1, step_size)

    for alpha in amounts:
        interpolated_latent_code = linear_interpolate(code1, code2, alpha)
        images = generate_image_from_z(G, interpolated_latent_code)
        interp_latent_image = Image.fromarray(images[0].cpu().numpy()).resize((256, 256))
        frame = get_concat_h(img1, interp_latent_image)
        frame = get_concat_h(frame, img2)
        all_imgs.append(frame)

    save_name = outdir + '/latent_space_traversal.gif'
    all_imgs[0].save(save_name, save_all=True, append_images=all_imgs[1:], duration=1000 / 20, loop=0)

if __name__=='__main__':
    modelpath = 'stylegan3-r-ffhqu-256.pkl'
    outdir = 'out'
    G = load_model(modelpath)
    image1, latent_code1 = generate_image_random(G, 1, outdir)
    image2, latent_code2 = generate_image_random(G, 1234, outdir)
    make_latent_interp_animation(G, latent_code1, latent_code2, image1, image2, num_interps=200, outdir=outdir)