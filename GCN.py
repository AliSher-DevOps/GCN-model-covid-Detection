
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from scipy.spatial.distance import cdist
torch.cuda.empty_cache()


class BorisNet(nn.Module):
    def __init__(self):
        super(BorisNet, self).__init__()
        self.fc = nn.Linear(4096, 10, bias=True)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))


class BorisConvNet(nn.Module):
    def __init__(self):
        super(BorisConvNet, self).__init__()
        self.conv = nn.Conv2d(1, 10, 28, stride=1, padding=14)
        self.fc = nn.Linear(4 * 4 * 10, 10, bias=False)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = F.max_pool2d(x, 7)
        return self.fc(x.view(x.size(0), -1))

class BorisGraphNet(nn.Module):
    def __init__(self,  pred_edge=True):
        super(BorisGraphNet, self).__init__()
        self.pred_edge = pred_edge
        N = ISZ ** 2
        self.fc = nn.Linear(N, 10, bias=False)
        if pred_edge:
            col, row = np.meshgrid(np.arange(ISZ), np.arange(ISZ))
            coord = np.stack((col, row), axis=2).reshape(-1, 2)
            coord = (coord - np.mean(coord, axis=0)) / (np.std(coord, axis=0) + 1e-5)
            coord = torch.from_numpy(coord).float()  # 784,2
            coord = torch.cat((coord.unsqueeze(0).repeat(N, 1,  1),
                                    coord.unsqueeze(1).repeat(1, N, 1)), dim=2)
            #coord = torch.abs(coord[:, :, [0, 1]] - coord[:, :, [2, 3]])
            self.pred_edge_fc = nn.Sequential(nn.Linear(4, 64),
                                              nn.ReLU(),
                                              nn.Linear(64, 1),
                                              nn.Tanh())
            self.register_buffer('coord', coord)
        else:
            # precompute adjacency matrix before training
            A = self.precompute_adjacency_images(ISZ)
            self.register_buffer('A', A)


    @staticmethod
    def precompute_adjacency_images(img_size):
        col, row = np.meshgrid(np.arange(img_size), np.arange(ISZ))
        coord = np.stack((col, row), axis=2).reshape(-1, 2) / ISZ
        dist = cdist(coord, coord)  
        sigma = 0.05 * np.pi
        A = np.exp(- dist / sigma ** 2)
            
        A[A < 0.01] = 0
        A = torch.from_numpy(A).float()
        D = A.sum(1)  # nodes degree (N,)
        D_hat = (D + 1e-5) ** (-0.5)
        A_hat = D_hat.view(-1, 1) * A * D_hat.view(1, -1)  # N,N
        A_hat[A_hat > 0.0001] = A_hat[A_hat > 0.0001] - 0.2

        print(A_hat[:10, :10])
        return A_hat

    def forward(self, x):
        B = x.size(0)
        if self.pred_edge:
            self.A = self.pred_edge_fc(self.coord).squeeze()
        # print ("SIZE OF A",self.A.size())
        # print ("SIZE OF A",x.size())
        avg_neighbor_features = (torch.bmm(self.A.unsqueeze(0).expand(B, -1, -1),
                                 x.view(B, -1, 1)).view(B, -1))
        return self.fc(avg_neighbor_features)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


def main():
      
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(description='GCN Model')
    parser.add_argument('--model', type=str, default='graph', choices=['GCN', 'graph', 'conv'],
                        help='model to use for training (default: GCN)')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=10,
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--pred_edge', action='store_true', default=True,
                        help='predict edges instead of using predefined ones')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=200,
                        help='how many batches to wait before logging training status')
    global ISZ
    ISZ = 1024
    args = parser.parse_args()
    use_cuda = True
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.Resize((ISZ,ISZ)),                                    transforms.ToTensor(),                                    transforms.Normalize([0.5],[0.5])                                    ])
    dataset = datasets.ImageFolder(PATHtoDir,transform = transform)
    dataset_len = len(dataset)
    print("LEN OF DATASET",dataset_len)
    train_len, test_len = int(dataset_len/2), int(dataset_len/2)
    print(train_len)
    print(test_len)
    train_set,test_set = torch.utils.data.random_split(dataset,[train_len,test_len])
    batch_size = 1
    train_loader = torch.utils.data.DataLoader(dataset=train_set, shuffle=True, batch_size= batch_size)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, shuffle=True, batch_size= batch_size)
    model = BorisGraphNet(pred_edge=args.pred_edge)
    model.to(device)
    print(model)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-1 if args.model == 'conv' else 1e-4)
    print('number of trainable parameters: %d' %
          np.sum([np.prod(p.size()) if p.requires_grad else 0 for p in model.parameters()]))

    for epoch in range(epochT):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)


global PATHtoDir,epochT
#replace this with dataset path
PATHtoDir = r'C:\Users\92344\Downloads\COVID CTSCAN'
epochT = 500

if __name__ == '__main__':
    main()





