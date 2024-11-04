import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import pdb
import itertools

def weight_init(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)
        m.bias.data.zero_()
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)

    if isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)

class deconv2x2(nn.Module):
    def __init__(self, in_c, out_c, k=4, s=2, p=1):
        super(deconv2x2, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels=in_c, out_channels=out_c, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
    
    def forward(self, input):
        x = self.deconv(input)
        x = self.bn(x)
        x = self.lrelu(x)
        return x

class Efficientdecoder1(nn.Module):
    def __init__(self, in_dim=1280):
        super(Efficientdecoder1, self).__init__()
        self.up = nn.ConvTranspose2d(in_dim, in_dim, kernel_size=2, stride=1, padding=0)
        self.deconv1 = deconv2x2(in_dim, 640)
        self.deconv2 = deconv2x2(640, 320, k=3)
        self.deconv3 = deconv2x2(320, 192)
        self.deconv4 = deconv2x2(192, 112)
        self.deconv5 = deconv2x2(112, 80)
        self.deconv6 = deconv2x2(80, 40)
        self.deconv7 = deconv2x2(40, 3)

        self.apply(weight_init)

    
    def forward(self, latent_code):
        # latent_code : [B x 1280]
        m1 = self.up(latent_code.reshape(latent_code.shape[0], -1, 1, 1))
        # m1 : [B x 1280 x 2 x 2]
        m2 = self.deconv1(m1)
        # m2 : [B x 640 x 4 x 4]
        m3 = self.deconv2(m2)
        # m3 : [B x 320 x 7 x 7]
        m4 = self.deconv3(m3)
        # m4 : [B x 192 x 14 x 14]
        m5 = self.deconv4(m4)
        # m5 : [B x 112 x 28 x 28]
        m6 = self.deconv5(m5)
        # m6 : [B x 80 x 56 x 56]
        m7 = self.deconv6(m6)
        # m7 : [B x 40 x 112 x 112]
        m8 = self.deconv7(m7)
        # m8 : [B x 3 x 224 x 224]
        # pdb.set_trace()
        return torch.tanh(m8)

if __name__=='__main__':
    model = Efficientdecoder1()
    # print(summary(model, (1, 1280), device='cpu'))
    x = torch.randn(16, 1280)
    result = model(x)
    print(result.shape)
    