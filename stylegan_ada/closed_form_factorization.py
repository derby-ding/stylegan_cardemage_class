import argparse
import dnnlib
import torch
import legacy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="factor.pt", help="name of the result factor file")
    parser.add_argument("--ckpt", type=str, default='cifar10.pkl', help="name of the model checkpoint")
    args = parser.parse_args()

    # ckpt = torch.load(args.ckpt)
    with dnnlib.util.open_url(args.ckpt) as f:
        ckpt = legacy.load_network_pkl(f)##['G_ema'].to(device)  # type: ignore
    print('check ckpt', ckpt['G'], type(ckpt['G']))
    modulate = {
        k: v
        for k, v in ckpt["G_ema"].items()
        if "modulation" in k and "to_rgbs" not in k and "weight" in k
    }

    weight_mat = []
    for k, v in modulate.items():
        weight_mat.append(v)

    W = torch.cat(weight_mat, 0)
    eigvec = torch.svd(W).V.to("cpu")

    torch.save({"ckpt": args.ckpt, "eigvec": eigvec}, args.out)

