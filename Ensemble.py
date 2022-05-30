
from __future__ import print_function
import torch
import torch.nn as nn
import argparse
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import f1_score
from torchmetrics import F1Score,ConfusionMatrix,Precision

def double_conv(in_c,out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=1),
        nn.ReLU(inplace=True),        
        )
    return conv

def crop_img(tensor,target_tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta =delta // 2
    return tensor[:,:,delta:tensor_size-delta,delta:tensor_size-delta]

class UNet(nn.Module):
    def __init__(self):
        super(UNet,self).__init__()
        # "Kernel size can't be greater than actual input size" change kernel and stride to 1  
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=1,stride=1)
        self.down_conv_1 = double_conv(1,64)
        self.down_conv_2 = double_conv(64,128)
        self.down_conv_3 = double_conv(128,256)
        self.down_conv_4 = double_conv(256,512)
        self.down_conv_5 = double_conv(512,1024)
        
        self.up_trans_1 = nn.ConvTranspose2d(in_channels=1024,
                                             out_channels=512,
                                             kernel_size=1,
                                             stride=1)
        
        self.up_conv_1 = double_conv(1024, 512)
        
        self.up_trans_2 = nn.ConvTranspose2d(in_channels=512,
                                             out_channels=256,
                                             kernel_size=1,
                                             stride=1)
        self.up_conv_2 = double_conv(512, 256)

        
        self.up_trans_3 = nn.ConvTranspose2d(in_channels=256,
                                             out_channels=128,
                                             kernel_size=1,
                                             stride=1)
        self.up_conv_3 = double_conv(256, 128)

        
        self.up_trans_4 = nn.ConvTranspose2d(in_channels=128,
                                             out_channels=64,
                                             kernel_size=1,
                                             stride=1)
        self.up_conv_4 = double_conv(128, 64)
        
        #change out channel based on no of outputs
        self.out = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1)
        
        self.fc1 = nn.Linear(128, 64)  #388 x 64
        self.relu_1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 2)

    
    def forward(self,image):
        #encoder
        x1 = self.down_conv_1(image)
        # print(x1.size())
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv_2(x2)
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv_3(x4)
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv_4(x6)
        x8 = self.max_pool_2x2(x7)
        x9 = self.down_conv_5(x8)

        #decoder
        x = self.up_trans_1(x9)
        # print(x.size())
        # print(x7.size())
        
        y = crop_img(x7, x)
        # print(y.size())
        
        x = self.up_conv_1(torch.cat([x,y],1))
        x = self.up_trans_2(x)
        y = crop_img(x5, x)
        x = self.up_conv_2(torch.cat([x,y],1))
        x = self.up_trans_3(x)
        y = crop_img(x3, x)
        x = self.up_conv_3(torch.cat([x,y],1))
        
        x = self.up_trans_4(x)
        y = crop_img(x1, x)
        x = self.up_conv_4(torch.cat([x,y],1))
        x = self.out(x)
        x=torch.flatten(x,1)
        
        # print('****')
        # print(x.size())
        
        x = self.fc1(x)
        x = self.relu_1(x)
        x = self.fc2(x)
       
        
        # return x
        
        output = F.log_softmax(x, dim=1)
        return output


class GraphConvNet(nn.Module):
    def __init__(self,  pred_edge=True):
        super(GraphConvNet, self).__init__()
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
        avg_neighbor_features = (torch.bmm(self.A.unsqueeze(0).expand(B, -1, -1),
                                 x.view(B, -1, 1)).view(B, -1))
        return self.fc(avg_neighbor_features)

class MyEnsemble(nn.Module):
    def __init__(self,model_GCN,model_UNET, nb_classes=2):
        super(MyEnsemble, self).__init__()
        self.gcn = model_GCN
        self.unet = model_UNET
        self.relu_1 = nn.ReLU()
        self.classifier = nn.Linear(12, 2)
        
    def forward(self,img):
        x1 = self.gcn(img)
        x2 = self.unet(img)
        x = torch.cat((x1,x2), dim=1)
        x = self.relu_1(x)
        x= self.classifier(x)
        
        return x

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
            preds = model(data)
            f1(preds,target)
            precision(preds,target)
            confmat(preds, target)
            
            

    test_loss /= len(test_loader.dataset)

    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

def main():
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(description='GCN+UNET')
    parser.add_argument('--model', type=str, default='graph', choices=[ 'graph', 'conv'],
                        help='model to use for training (default: fc)')
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
    global ISZ,f1,precision,confmat
    f1 = F1Score(num_classes = 2)
    precision = Precision(average='macro',num_classes=2)
    confmat = ConfusionMatrix(num_classes=2)
    ISZ = 8
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),                          transforms.Resize((ISZ,ISZ)),                                    transforms.ToTensor(),                                    transforms.Normalize([0.5],[0.5])                                    ])
    dataset = datasets.ImageFolder(PATHtoDir,transform = transform)
    dataset_len = len(dataset)
    print("LEN OF DATASET",dataset_len)
    train_len, test_len = int(dataset_len/2), int(dataset_len/2)
    train_set,test_set = torch.utils.data.random_split(dataset,[train_len,test_len])
    batch_size = 1
    train_loader = torch.utils.data.DataLoader(dataset=train_set, shuffle=True, batch_size= batch_size)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, shuffle=True, batch_size= batch_size)
    modelG=GraphConvNet(pred_edge=args.pred_edge)
    modelU=UNet()
    
    model = MyEnsemble(modelG,modelU)
    model.to(device)
    print(model)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-1 if args.model == 'conv' else 1e-4)
    print('number of trainable parameters: %d' %
          np.sum([np.prod(p.size()) if p.requires_grad else 0 for p in model.parameters()]))

    for epoch in range(epochT):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
    
    print("F1 score :",f1.compute())
    print('Precision :',precision.compute())
    print('Confusion Matrix:',confmat.compute())





global PATHtoDir,epochT
PATHtoDir = r'C:\Users\PC 5\Downloads\COVID CTSCAN'
epochT = 100

if __name__ == '__main__':
    main()


        
        
        
        