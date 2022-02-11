import torch
import torch.nn as nn

def Conv3x3(in_channels, out_channels,slope = 0.1):
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size = 3, stride = 1, padding = 1, bias = False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(slope)
    )

def convStep(in_channels,out_channels,slope = 0.1):
    return nn.Sequential(
        Conv3x3(in_channels,out_channels,slope),
        Conv3x3(out_channels,out_channels,slope)
    )

class Unet(nn.Module):
    def __init__(self):
        super(Unet,self).__init__()
        channels = [3,64,128,256,512,1024]
        self.downSteps = nn.ModuleList([Conv3x3(channels[i],channels[i+1]) for i in range(5)])
        self.upSteps = nn.ModuleList([Conv3x3(channels[-i],channels[-(i+1)]) for i in range(1,5)])
        self.downSample = nn.MaxPool2d(2)
        self.upSamples = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(channels[-i],channels[-(i+1)],kernel_size = 2, stride = 2,bias = False),
                nn.BatchNorm2d(channels[-(i+1)]),
                nn.LeakyReLU(0.1)
            ) for i in range(1,5)])
        self.output = nn.Sequential(
            nn.Conv2d(channels[1],6,kernel_size = 1, stride = 1, padding = 0, bias = False),
            nn.ReLU()
        )

    def forward(self,x):
        xs = []
        for i in range(4):
            x = self.downSteps[i](x)
            xs.append(x)
            x = self.downSample(x)
        x = self.downSteps[4](x)
        for i in range(4):
            x = self.upSamples[i](x)
            x = torch.cat((x,xs[-(i+1)]),dim = 1)
            x = self.upSteps[i](x)
        return self.output(x)

if '__main__' == __name__:
    model = Unet()
#     print(model.parameters)
    model.cuda()
    x = torch.rand(4,1,512,512).to('cuda')
    out = model(x)
    print(out.shape)
