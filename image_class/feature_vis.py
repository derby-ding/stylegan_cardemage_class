import timm
import torch
from timm.data import ImageDataset, create_loader, resolve_data_config
'''提取图像特征，'''
def get_feat(modelname, checkpoint, loader):
    ####efficientnet_b0 -- b7 inception_v3 v4 resnet18 51 101 mobilenet
    model = timm.create_model(modelname, checkpoint_path=checkpoint)
    # imgs = torch.randn(2, 3, 224, 224)
    # feats = model.forward_features(imgs)
    # print('check shape ', feats.shape)
    model.cuda()
    model.eval()
    tfeat = []
    tlabl = []
    with torch.no_grad():
        for batch_idx, (input, labels) in enumerate(loader):
            input = input.cuda()
            feats = model.forward_features(input)
            tfeat.append(feats)
            tlabl.append(labels)
    tfeat = torch.stack(tfeat).squeeze()
    tlabl = torch.stack(labels).squeeze()
    print('check tfeat tlabl shape', tfeat.shape, tlabl.shape)

    nfeat = tfeat.detach().cpu().numpy()
    nlabl = tlabl.detach().cpu().numpy()
    return nfeat, nlabl
'''可视化aug前后的图像特征分布，图像特征使用efficientnet提取，并降维到2维'''
def vis_aug_2d(oridir, augdir):
    loader = create_loader(ImageDataset(oridir),
                           input_size=256,
                           batch_size=1,
                           use_prefetcher=True,
                           interpolation='bicubic',  ###bilinear bicubic
                           mean=[0.5, 0.5, 0.5],
                           std=[0.5, 0.5, 0.5],
                           num_workers=2,
                           crop_pct=1.0)
    modelname = 'efficientnet_b2'
    checkpoint = 'output/efficientnet_b2/model_best.pth.tar'
    orifeat, orilab = get_feat(modelname, checkpoint, loader)
    loader = create_loader(ImageDataset(augdir),
                           input_size=256,
                           batch_size=1,
                           use_prefetcher=True,
                           interpolation='bicubic',  ###bilinear bicubic
                           mean=[0.5, 0.5, 0.5],
                           std=[0.5, 0.5, 0.5],
                           num_workers=2,
                           crop_pct=1.0)
    augfeat, auglab = get_feat(modelname, checkpoint, loader)

    # umap_vis(orifeat, auglab, tx=augfeat, savename='augvis.png')
    reducer = umap.UMAP(random_state=42, metric='cosine')
    samnum = orifeat.shape[0]
    if orifeat.ndim > 2:
        orifeat = orifeat.reshape(samnum, -1)
        reducer.fit(orifeat)
    else:
        reducer.fit(orifeat)
    dx = reducer.transform(orifeat)
    X_umap = np.vstack((dx.T, orilab)).T
    df_ump = pd.DataFrame(X_umap, columns=['Dim1', 'Dim2', 'class'])
    df_ump[['class']] = df_ump[['class']].astype(int)

    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=df_ump, hue='class', x='Dim1', y='Dim2', palette='deep')
    plt.savefig('orimap.png')

    dx = reducer.transform(augfeat)
    X_umap = np.vstack((dx.T, auglab)).T
    df_ump = pd.DataFrame(X_umap, columns=['Dim1', 'Dim2', 'class'])
    df_ump[['class']] = df_ump[['class']].astype(int)

    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=df_ump, hue='class', x='Dim1', y='Dim2', palette='deep')
    plt.savefig('augmap.png')

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import umap
'''umap降维，X为映射集，tx为测试集，umap可以用映射集构建映射，然后将新数据映射到已有映射中'''
def umap_vis(X, y, tx=None, savename='umap.png'):
    reducer = umap.UMAP(random_state=42)
    samnum = X.shape[0]
    if X.ndim > 2:
        X = X.reshape(samnum, -1)
    reducer.fit(X)
    if tx is not None:
        dx = reducer.transform(tx)
    else:
        dx = reducer.transform(X)

    X_umap = np.vstack((dx.T, y)).T
    df_ump = pd.DataFrame(X_umap, columns=['Dim1', 'Dim2', 'class'])

    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=df_ump, hue='class', x='Dim1', y='Dim2')
    plt.savefig(savename)

if __name__=='__main__':
    # imgs = torch.randn(2, 3, 224, 224)
    # loader = create_loader(ImageDataset('../datasets/clsdema/cls8/train'),
    #     input_size=256,
    #     batch_size=1,
    #     use_prefetcher=True,
    #     interpolation='bicubic',###bilinear bicubic
    #     mean=[0.5, 0.5, 0.5],
    #     std=[0.5, 0.5, 0.5],
    #     num_workers=2,
    #     crop_pct=1.0)
    # model_names = timm.list_models('*efficient*')
    # modelname = 'efficientnet_b2'
    # checkpoint = 'model_best.pth.tar'
    # get_feat(modelname, checkpoint, loader)
    oridir = '../dataset/comaug/train'
    augdir = '../dataset/comaug/aug'
    vis_aug_2d(oridir, augdir)

