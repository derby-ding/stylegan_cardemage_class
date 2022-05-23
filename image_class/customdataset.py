'''高维多标签分类，图像要素提取'''
import torch, os, re
import numpy as np
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from os import listdir
from os.path import isfile, join
import torchvision
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import difflib
from PIL import Image
catls = ['tail light', "tail bumper", "back wheel", "door", "plate", "front wheel", "front light", "front bumper", "deformation", "fall", 'person', 'break', 'scratch', 'other']
cat2id = {cat:i for i, cat in enumerate(catls)}

'''读obdet格式的txt，转为imagenamels, labells的格式'''
def load_labeledjson(labeldir, imagdir):
    imnames = []
    labels = []

    '''遍历文件，抽取labels'''
    files = os.listdir(labeldir)
    for file in files:
        filename = os.path.join(labeldir, file)
        if not os.path.isdir(filename) and file.endswith('.txt'):
            tlab = ''
            with open(filename, 'r') as fi:
                lines = fi.readlines()
                for i, line in enumerate(lines):
                    if i==0:
                        tlab = str(re.split(' ', line)[0])
                    else:
                        tlab += '|'
                        tlab += str(re.split(' ', line)[0])
            labels.append(tlab)
            imfile = re.sub('.txt', '.jpg', file)
            imnames.append(os.path.join(imagdir, imfile))
    return imnames, labels

def default_loader(path):
    return Image.open(path).convert('RGB')
'''输入imgls图像文件名，labls形如0，数字表示类型，num_cls表示总种类'''
class MyDataset(Dataset):
    def __init__(self, imgls, labls, num_cls, transform=None, target_transform=None, loader=default_loader, multi=False):

        self.imgs = imgls
        self.labs = labls
        self.num_cls = num_cls
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.multi = multi
    def __getitem__(self, index):
        fn = self.imgs[index]
        label = self.labs[index]
        if self.multi:
            clses = re.split('[|]', label)###多标签label以|分割
            label = [0 for _ in range(self.num_cls)]  ###等长序列表示标签
            for lab in clses:
                label[int(lab)] = 1

        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        seqlab = torch.tensor(label, dtype=torch.float)
        return img, seqlab

    def __len__(self):
        return len(self.imgs)


'''输入训练modle, 保存路径'''
def train_eval(train_imgs,train_labels, test_imgs, test_labels, outmodeldir,num_labels):
    train_augmentation = torchvision.transforms.Compose([torchvision.transforms.Resize((480, 480)),
                                                         torchvision.transforms.RandomHorizontalFlip(),
                                                         torchvision.transforms.ToTensor(),
                                                         torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                         ])
    train_data = MyDataset(imgls=train_imgs, labls=train_labels, num_cls=num_labels, transform=train_augmentation)
    test_data = MyDataset(imgls=test_imgs, labls=test_labels, num_cls=num_labels, transform=train_augmentation)
    # data_loader = DataLoader(train_data, batch_size=16, shuffle=True)  ###num_workers线程
    do_train = True
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = effnetv2_l(num_classes=num_labels)
    if do_train:
        model.to(device)
        model.train()
        train_loader = DataLoader(train_data, batch_size=12, shuffle=True)
        optim = AdamW(model.parameters(), lr=1e-4, weight_decay=3e-6)
        # Weights = torch.FloatTensor(Weights).to(device)
        # lossfct = FocalLoss1(weight=Weights)
        loss_fct = torch.nn.BCEWithLogitsLoss()

        for epoch in range(100):
            tloss = 0.0
            tacc = 0
            for bimgs, blabs in train_loader:
                optim.zero_grad()
                input_ids = bimgs.to(device)
                labels = blabs.to(device)
                outputs = model(input_ids)
                # pred = torch.argmax(outputs[1], dim=1)
                # print('check pred labels', outputs, '\n',labels)
                # # loss = F.cross_entropy(outputs[1], labels, weight=Weights)
                # acc = ((pred == labels).sum()).item()
                loss = loss_fct(outputs.view(-1, num_labels), labels.view(-1, num_labels))
                tloss += loss.item()
                # tacc += acc
                loss.backward()
                optim.step()
            print('check epoch', epoch, 'loss', tloss)#, 'acc 百分值', tacc / tnum * 100)

        torch.save(model, outmodeldir+'/multilab.pt')  ##model.state_dict仅保存权重,model保存整体

    ##testing
    # newmodel = BertForSequenceClassification.from_pretrained(outmodeldir, config=config)
    model = torch.load(outmodeldir+'/multilab.pt')
    model.to(device)
    model.eval()
    logit_preds, true_labels, pred_labels = [], [], []
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids)
        b_logit_pred = outputs
        pred_label = torch.sigmoid(b_logit_pred).detach().cpu().numpy()
        b_labels = labels.to('cpu').numpy()

        true_labels.append(b_labels)
        pred_labels.append(pred_label)

    pred_labels = [item for sublist in pred_labels for item in sublist]###全展开
    true_labels = [item for sublist in true_labels for item in sublist]

    # Calculate Accuracy
    threshold = 0.50
    pred_bools = [pl > threshold for pl in pred_labels]
    true_bools = [tl == 1 for tl in true_labels]
    val_f1_accuracy = metrics.f1_score(true_bools, pred_bools, average='micro') * 100
    val_flat_accuracy = metrics.accuracy_score(true_bools, pred_bools) * 100
    print('check eval f1 score', val_f1_accuracy, val_flat_accuracy)

'''载入model，并对infile样本进行判定，给出概率，方便复检'''
def pred_augment(modeldir, txts, labs, id2lab):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torch.load(modeldir+'/multilab.pt', map_location=device)
    tokenizer = BertTokenizer.from_pretrained(modeldir)
    test_encodings = tokenizer(txts, truncation=True, padding='max_length', max_length=maxlen)
    test_dataset = MultilabDataset(test_encodings, labs)
    model.to(device)
    model.eval()
    predictlabs = []
    predicts = []
    test_loader = DataLoader(test_dataset, batch_size=12, shuffle=False)
    for batch in tqdm(test_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        predictlabs.extend(probs)
    for probs in predictlabs:
        prdl = [i for i, val in enumerate(probs) if val > 0.5]
        prds = '|'.join([id2lab.get(l) for l in prdl])
        # print('check prd lab', prds)
        predicts.append(prds)  ##sigmoid

    return predicts
'''载入类似imagenet格式的image dir'''
def load_imgdir(datadir):
    imgnames = []
    labels = []

    for i, f in enumerate(listdir(datadir)):
        if os.path.isdir(join(datadir, f)):
            catdir = join(datadir, f)
            for fi in listdir(catdir):
                if isfile(join(catdir, fi)) and fi.split('.')[1] == 'png':
                    imgnames.append(join(catdir, fi))
                    labels.append(i)
    # for fi in filenames:
    #     fpath = fi.split('[/|\]')
    #     imgnames.append(fpath[1])
    #     labels.append(labdict.get(fpath[0]))###转化成int
    return imgnames, labels
'''多标签分类任务'''
def get_dataset(indir, imsize=(480, 480), mode='OD'):
    train_augmentation = torchvision.transforms.Compose([torchvision.transforms.Resize(imsize),
                                                         torchvision.transforms.RandomHorizontalFlip(),
                                                         torchvision.transforms.ToTensor(),
                                                         torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                                                          [0.229, 0.224, 0.225])
                                                         ])
    if mode == 'OD':
        imagdir = indir + 'images/train2017'
        labldir = indir + 'labels/train2017'
        imgnames, labels = load_labeledjson(labldir, imagdir)
    else:
        imgnames, labels = load_imgdir(indir)
    num_labels = len(set(labels))
    train_imgs, test_imgs, train_labels, test_labels = train_test_split(imgnames, labels,
                                                                        test_size=0.2)  ###随机选取, shuffle=True
    train_data = MyDataset(imgls=train_imgs, labls=train_labels, num_cls=num_labels, transform=train_augmentation)
    test_data = MyDataset(imgls=test_imgs, labls=test_labels, num_cls=num_labels, transform=train_augmentation)
    return train_data, test_data

if __name__=='__main__':
    indir = '/data/dingkai/carinsure_cv/datasets/cardema/'
    ims, lbs = load_imgdir('F:\carinsure_cv\datasets\clsdema\openori')
    print('check ims', ims[:5], lbs[:5])


