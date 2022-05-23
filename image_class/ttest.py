from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torch.utils.data import DataLoader

path = '../datasets/clsdema/cls8/train'
transform = T.Compose([T.Resize((150, 200)),T.ToTensor()])

dataset = ImageFolder(root=path, transform=transform)

dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
####测试image和labels
for batch_number, (images, labels) in enumerate(dataloader):
    print(batch_number, labels)
    # print(images)