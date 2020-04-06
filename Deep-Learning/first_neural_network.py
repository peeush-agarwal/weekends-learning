import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Define a neural network 
# 1. Input image : 32 x 32 x 1 (W x H x Channels)
# 2. Conv2d layer: 28 x 28 x 6
# 3. Conv2d layer: 10 x 10 x 16
# 4. Linear layer: 120
# 5. Linear layer: 84
# 6. Output layer: 10

class FirstNN(nn.Module):
    def __init__(self):
        super(FirstNN, self).__init__()

        # nn.Conv2d will take in a 4D Tensor of nSamples x nChannels x Height x Width
        self.conv1 = nn.Conv2d(1, 6, 3)     # Weights: 32x32x1 => 28x28x6
        self.conv2 = nn.Conv2d(6, 16, 3)    # Weights: 28x28x6 => 10x10x16
        self.fc1 = nn.Linear(16*6*6, 120)   # (6*6) => Image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)  # 2 => (2,2) max pool layer
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)

        return out

    def num_features(self, x): # [batch_size, dim1, dim2, dim3]
        size = x.size() # (batch_size, ...)
        size = size[1:] # (dim1, ...)
        num_feat = 1
        for s in size:
            num_feat *= s
        return num_feat     # dim1 * dim2 * dim3

def train(net, criterion, optimizer, input, target):
    optimizer.zero_grad()   # zero the gradient buffers
    output = net(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()    # Does the update

if __name__ == "__main__":
    net = FirstNN()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    
    x = torch.rand(1, 1, 32, 32) # Batch of 1 of image: 1 x 32 x 32
    target = torch.rand(10)         # Random target variable
    target = target.view(1, -1)     # Reshape to match size of output
    
    train(net, criterion, optimizer, x, target)
    pass

    # print(net)

    # print('Neural network parameters weights:')
    # params = list(net.parameters())
    # for param in params:
    #     print(param.size())

    # # Random example of forward propogation
    # out = net(x)
    # print(out)

    # # Compute loss
    
    
    # loss = criterion(out, target)   # Compute loss
    # print(f'Loss: {loss}')
    # print(loss.grad_fn)             # MSELoss
    # print(loss.grad_fn.next_functions[0][0])    # LInear
    # print(loss.grad_fn.next_functions[0][0].next_functions[0][0]) # RELU


    # net.zero_grad()     # zeroes the gradient buffers of all parameters

    # print('conv1.bias.grad before backward')
    # print(net.conv1.bias.grad)

    # loss.backward()

    # print('conv1.bias.grad after backward')
    # print(net.conv1.bias.grad)

    

    # pass
