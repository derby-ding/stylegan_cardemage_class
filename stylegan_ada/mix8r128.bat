####cp -r ../datasets/augcls4/train ../datasets/mixcls4/train
python style_mixing.py --outdir=../datasets/mixcls8/train/00 --rows=0-100 --cols=101-143 --network=runs/car/00029-cls8gan128-cond-mirror-cifar-kimg10000-batch128-bgcfnc/network-010000.pkl;
python style_mixing.py --outdir=../datasets/mixcls8/train/01 --rows=144-220 --cols=221-278 --network=runs/car/00029-cls8gan128-cond-mirror-cifar-kimg10000-batch128-bgcfnc/network-010000.pkl;
python style_mixing.py --outdir=../datasets/mixcls8/train/02 --rows=279-350 --cols=351-397 --network=runs/car/00029-cls8gan128-cond-mirror-cifar-kimg10000-batch128-bgcfnc/network-010000.pkl;
python style_mixing.py --outdir=../datasets/mixcls8/train/03 --rows=398-498 --cols=788-878 --network=runs/car/00029-cls8gan128-cond-mirror-cifar-kimg10000-batch128-bgcfnc/network-010000.pkl;
python style_mixing.py --outdir=../datasets/mixcls8/train/04 --rows=879-950 --cols=951-991 --network=runs/car/00029-cls8gan128-cond-mirror-cifar-kimg10000-batch128-bgcfnc/network-010000.pkl;
python style_mixing.py --outdir=../datasets/mixcls8/train/05 --rows=992-1060 --cols=1061-1108 --network=runs/car/00029-cls8gan128-cond-mirror-cifar-kimg10000-batch128-bgcfnc/network-010000.pkl;
python style_mixing.py --outdir=../datasets/mixcls8/train/06 --rows=1109-1180 --cols=1181-1228 --network=runs/car/00029-cls8gan128-cond-mirror-cifar-kimg10000-batch128-bgcfnc/network-010000.pkl;
python style_mixing.py --outdir=../datasets/mixcls8/train/07 --rows=1229-1330 --cols=1331-1396 --network=runs/car/00029-cls8gan128-cond-mirror-cifar-kimg10000-batch128-bgcfnc/network-010000.pkl;