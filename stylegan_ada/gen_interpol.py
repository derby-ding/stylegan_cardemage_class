import dnnlib, os, re
import legacy
import numpy as np
import click
import torch
import PIL
from typing import List
from PIL import Image
device = torch.device('cuda')

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

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--rows', 'row_seeds', type=num_range, help='Random seeds to use for image rows', required=True)
@click.option('--cols', 'col_seeds', type=num_range, help='Random seeds to use for image columns', required=True)
@click.option('--styles', 'col_styles', type=num_range, help='Style layer range', default='0-6', show_default=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--outdir', type=str, required=True)
def generate_interp(
    network_pkl: str,
    row_seeds: List[int],
    col_seeds: List[int],
    col_styles: List[int],
    truncation_psi: float,
    noise_mode: str,
    outdir: str
):
    """Generate images by interpolation. light4tr 0-174 0 front break 175-291 1 tail break 292-450 2 tail 451-569 3 front
    Examples:
    python gen_interpol.py --outdir=out --rows=85,100,75,458,1500 --cols=55,821,1789,293 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
    """
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    os.makedirs(outdir, exist_ok=True)

    print('Generating W vectors...')
    all_seeds = list(set(row_seeds + col_seeds))
    all_z = np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in all_seeds])
    all_w = G.mapping(torch.from_numpy(all_z).to(device), None)
    w_avg = G.mapping.w_avg
    all_w = w_avg + (all_w - w_avg) * truncation_psi
    w_dict = {seed: w for seed, w in zip(all_seeds, list(all_w))}

    print('Generating images...')
    all_images = G.synthesis(all_w, noise_mode=noise_mode)
    all_images = (all_images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
    image_dict = {(seed, seed): image for seed, image in zip(all_seeds, list(all_images))}

    print('Generating style-mixed images...')
    for row_seed in row_seeds:
        for col_seed in col_seeds:
            w = w_dict[row_seed].clone()
            w[col_styles] = w_dict[col_seed][col_styles]
            image = G.synthesis(w[np.newaxis], noise_mode=noise_mode)
            image = (image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            image_dict[(row_seed, col_seed)] = image[0].cpu().numpy()

    print('Saving images...')
    os.makedirs(outdir, exist_ok=True)
    for (row_seed, col_seed), image in image_dict.items():
        PIL.Image.fromarray(image, 'RGB').save(f'{outdir}/{row_seed}-{col_seed}.png')


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_interp() # pylint: disable=no-value-for-parameter