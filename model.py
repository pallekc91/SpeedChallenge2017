import torch.nn as nn
import torch.nn.functional as F


def normalize(x):
    return x / 255


class nvidiaModel(nn.Module):
    def __init__(self, inp_s=(3, 66, 220), batch_norm=False):
        super(nvidiaModel, self).__init__()
        self.batchNorm = batch_norm
        self.input_shape = inp_s
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)

        self.drop_out = nn.Dropout()

        self.conv4 = nn.Conv2d(48, 64, kernel_size=3)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3)

        self.lin1 = nn.Linear(64 * 1 * 20, 100)
        self.lin2 = nn.Linear(100, 50)
        self.lin3 = nn.Linear(50, 10)
        self.lin4 = nn.Linear(10, 1)

    def forward(self, x):
        x = x.permute(0,3,1,2)
        x = normalize(x)
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = self.drop_out(F.elu(self.conv3(x)))
        x = F.elu(self.conv4(x))
        x = F.elu(self.conv5(x))
        x = x.reshape(x.size(0), -1) 
        x = F.elu(self.lin1(x))
        x = F.elu(self.lin2(x))
        x = F.elu(self.lin3(x))
        x = self.lin4(x)
        return x


def get_nvidia_model(input_shape=(3, 66, 220)):
    return nvidiaModel(input_shape)
