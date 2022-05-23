'''基于预训练模型进行两图像插值混合， 包括训练集抽取图像插值，自定义图像插值等'''

import os, pickle
import numpy as np
# run_dir = 'out/test'
# snapshot_data = {'data': [np.random.randn(80, 100)]}
# # snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl')###cover
# snapshot_pkl = os.path.join(run_dir, f'network0.pkl')
# for i in range(10):
#     with open(snapshot_pkl, 'wb') as f:###覆盖
#         pickle.dump(snapshot_data, f)
import dnnlib
import legacy
import click
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
def generate_image_random(G, outdir, seed, trun=1, noise_mode='const'):
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

'''合并多帧'''
def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

'''插值动态图, code1和code2对应训练样本'''
def make_latent_interp_animation(G, code1, code2, img1, img2, num_interps, save_name):
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

    # save_name = outdir + '/latent_space_traversal.gif'
    all_imgs[0].save(save_name, save_all=True, append_images=all_imgs[1:], duration=1000 / 20, loop=0)

'''将新样本投影，或者说embedding到模型空间, sourcea表示源a为起点，对target图像进行拟合，输出为target图像的embedding和最后的拟合图像'''
from projector import project
# @click.command()####简易参数设置
# @click.option('--sourcea', help='Random seed', type=int, default=303, show_default=True)
def project_customim(modelpath, target_fname, sourcea=1):
    np.random.seed(sourcea)
    torch.manual_seed(sourcea)

    with dnnlib.util.open_url(modelpath) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore
    # print('check G dict', G)

    # Load target image.
    target_pil = PIL.Image.open(target_fname).convert('RGB')
    target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
    target_uint8 = np.array(target_pil, dtype=np.uint8)

    # Optimize projection.
    projected_w_steps = project(G, target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device),  # pylint: disable=not-callable
                        num_steps=800, device=device)

    # Save final projected frame and W vector.
    # target_pil.save(f'{outdir}/target.png')
    projected_w = projected_w_steps[-1]
    print('check projected_w ', projected_w.shape)
    synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
    synth_image = (synth_image + 1) * (255 / 2)
    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    synth_image = PIL.Image.fromarray(synth_image, 'RGB')
    # PIL.Image.fromarray(synth_image, 'RGB').save(f'{outdir}/proj.png')
    # G.mapping
    return synth_image, projected_w

'''插值，投影图像和抽样图像，可以抽样图像之间插值，投影图像之间插值，但不可以抽样和投影图像之间插值，why，
可以先将抽样图像投影，然后插值'''
def customsinglat_interp(G, codes1, codes2, img1, img2, num_interps, save_name):
    step_size = 1.0 / num_interps
    all_imgs = []
    amounts = np.arange(0, 1, step_size)
    # codes2 = code2.repeat([1, G.mapping.num_ws, 1])
    for alpha in amounts:
        interpolated_latent_code = linear_interpolate(codes1, codes2, alpha)

        # images = generate_image_from_z(G, interpolated_latent_code)
        synth_image = G.synthesis(interpolated_latent_code.unsqueeze(0), noise_mode='const')
        # print('check interpolated_latent_code', interpolated_latent_code.shape)
        # interp_latent_image = Image.fromarray(images[0].cpu().numpy()).resize((256, 256))
        synth_image = (synth_image + 1) * (255 / 2)
        synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        synth_image = PIL.Image.fromarray(synth_image, 'RGB')
        frame = get_concat_h(img1, synth_image)
        frame = get_concat_h(frame, img2)
        all_imgs.append(frame)

    # save_name = outdir + '/latent_space_traversal.gif'
    all_imgs[0].save(save_name, save_all=True, append_images=all_imgs[1:], duration=100 / 20, loop=0)

if __name__=='__main__':
    modelpath = 'stylegan3-r-ffhqu-256.pkl'
    outdir = 'out'
    ####训练集插值混合
    G = load_model(modelpath)
    # image1, latent_code1 = generate_image_random(G, outdir,  1)
    # image2, latent_code2 = generate_image_random(G, outdir, 1234)
    # make_latent_interp_animation(G, latent_code1, latent_code2, image1, image2, num_interps=200, save_name='out/sampsinterp.gif')

    ####自定义图像插值
    target_fname = 'out/1.png'
    target_fname2 = 'out/27.png'
    # target_pil = PIL.Image.open(target_fname).convert('RGB')
    image1, latent_code1 = project_customim(modelpath, target_fname)
    # image2, latent_code2 = generate_image_random(G, outdir, 1)
    image2, latent_code2 = project_customim(modelpath, target_fname2)
    customsinglat_interp(G, latent_code1, latent_code2, image1, image2, num_interps=20, save_name='out/cuslatint.gif')
    # print('check latent code shape', latent_code1.shape, latent_code2.shape)
    # make_latent_interp_animation(G, latent_code1[0], latent_code2, image1, image2, num_interps=200, save_name='out/custominterp0.gif')
    # image1 = generate_image_from_z(G, latent_code1[0])
    # make_latent_interp_animation(G, latent_code1[0], latent_code2, image1, image2, num_interps=200, save_name='out/custominterp1.gif')
    # make_latent_interp_animation(G, latent_code1[8], latent_code2, image1, image2, num_interps=200,
    #                              save_name='out/custominterp8.gif')