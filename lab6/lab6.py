# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import torch as pt
import torchvision as ptv
import torchvision.transforms as transforms
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

batch_size = 100
lr = 0.001
momentum = 0.9
epochs = 15
device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')

# %% [code]
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(outchannel),
            nn.PReLU(),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)
        
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# %% [code]
def load_data():
    # 标准化数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = ptv.datasets.CIFAR10(root='/kaggle/working/cifar10', train=True, transform=transform, download=True)
    test_dataset = ptv.datasets.CIFAR10(root='/kaggle/working/cifar10', train=False, transform=transform, download=True)

    # 建立数据集迭代器
    train_loader = pt.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = pt.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    print('size of train_dataset:', len(train_dataset), len(train_dataset[0][0]), len(train_dataset[0][0][0]), len(train_dataset[0][0][0][0]))
    print('size of test_dataset:', len(test_dataset), len(test_dataset[0][0]), len(test_dataset[0][0][0]), len(test_dataset[0][0][0][0]))
    
    return train_loader, test_loader

# %% [code]
class Model(object):
    def __init__(self, model, train_loader, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
    
    def train(self, optimizer, criterion):
        import time
        import random
        
        self.train_acc, self.train_loss = [], []
        self.test_acc, self.test_loss = [], []
        
        epoch = 0
        while True:
            epoch += 1
            start = time.perf_counter()
            
            tr_acc, tr_loss = [], []
            te_acc, te_loss = [], []
            
            for train_data in self.train_loader:
                features, labels = train_data
                features, labels = Variable(features).to(device), Variable(labels).to(device)
                self.model.train()
                optimizer.zero_grad()
                outputs = self.model(features)
                
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                tr_acc.append(float(pt.sum(pt.argmax(outputs, dim=1) == labels)) / batch_size)
                tr_loss.append(float(loss))
            
            for test_data in self.test_loader:
                features, labels = train_data
                features, labels = Variable(features).to(device), Variable(labels).to(device)
                self.model.eval()
                outputs = self.model(features)
                
                loss = criterion(outputs, labels)
                te_acc.append(float(pt.sum(pt.argmax(outputs, dim=1) == labels)) / batch_size)
                te_loss.append(float(loss))
                
            end = time.perf_counter()
            
            tr_acc = np.mean(tr_acc)
            te_acc = np.mean(te_acc)
            tr_loss = np.mean(tr_loss)
            te_loss = np.mean(te_loss)
            print('epcho[{}]: {:.2f}s'.format(epoch, end - start))
            print('\tTrain\tloss: {:.4f} acc: {:.4f}'.format(tr_loss, tr_acc))
            print('\tTest\tloss: {:.4f} acc: {:.4f}'.format(te_loss, te_acc))
            
            self.train_acc.append(tr_acc)
            self.train_loss.append(tr_loss)
            self.test_acc.append(te_acc)
            self.test_loss.append(te_loss)

            if te_acc > 0.905 and tr_acc > 0.955:
                pt.save(self.model.state_dict(), '/kaggle/working/ResNet')
                break
    
    
    def save(self, path):
        np.savez(path, train_acc=self.train_acc, train_loss=self.train_loss, test_acc=self.test_acc, test_loss=self.test_loss)
    
    
    def load(self, path):
        self.model.load_state_dict(pt.load(path))

    @staticmethod
    def draw(path):
        import matplotlib.pyplot as plt

        result = np.load(path, allow_pickle=bool)

        train_acc = result["train_acc"]
        train_loss = result["train_loss"]
        test_acc = result["test_acc"]
        test_loss = result["test_loss"]

        print(train_acc)
        print(train_loss)
        print(test_acc)
        print(test_loss)

        epochs = len(train_acc)

        ax = plt.subplot(1, 2, 1)
        ax.plot(train_acc, label='train', marker='.')
        ax.plot(test_acc, label='test', marker='.')
        plt.xlabel('epochs')
        plt.xlim(0, epochs)
        plt.ylabel('accuracy')
        plt.ylim(0, 1)
        plt.legend()

        ax=plt.subplot(1, 2, 2)
        ax.plot(train_loss, label='train', marker='.')
        ax.plot(test_loss, label='test', marker='.')
        plt.xlabel('epochs')
        plt.xlim(0, epochs)
        plt.ylabel('loss')
        plt.legend()

        plt.subplots_adjust(hspace=0.2, wspace=0.3)
        plt.savefig('/kaggle/working/result.pdf', dpi=700, bbox_inches='tight')
        plt.show()

# %% [code]
if __name__ == '__main__':
    if pt.cuda.device_count() > 0:
        import random
        devid = random.randint(0, pt.cuda.device_count() - 1)
        pt.cuda.set_device(devid)
        print('cuda device', devid)
    
    print('loading CIFAR10...')
    train_loader, test_loader = load_data()
    
    model = ResNet(ResidualBlock).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    
    print(model)
    print('start training...')
    trainer = Model(model, train_loader, test_loader)
    trainer.train(optimizer, criterion)
    
    print('saving results...')
    trainer.save('/kaggle/working/acc.npz')
    Model.draw('/kaggle/working/acc.npz')
    
    print('exit')
    exit(0)
