import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F
import numpy as np


class highPassEnsemble(nn.Module):
    def __init__(self, kernel_size=2):
        super(highPassEnsemble, self).__init__()
        self.kernel_size = kernel_size
        if kernel_size == 2:
            kernel_w = np.array([[0,  0, 0],
                                 [-1, 1, 0],
                                 [0,  0, 0]], dtype=np.float32)
            kernel_h = np.array([[0, -1, 0],
                                 [0,  1, 0],
                                 [0,  0, 0]], dtype=np.float32)
            self.conv_w = HighPassHard(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=3 // 2, dilation=1, groups=3, bias=False)
            self.conv_h = HighPassHard(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=3 // 2, dilation=1, groups=3, bias=False)
            self.conv_w.init_weight_highpass(kernel_w)
            self.conv_h.init_weight_highpass(kernel_h)
            print("HP-2 initialized")
        elif kernel_size == 3:
            # 3x3 high pass
            kernel = np.array([[-1, -1, -1],
                               [-1, 8, -1],
                               [-1, -1, -1]], dtype=np.float32)
            self.conv = HighPassHard(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=3 // 2, dilation=1, groups=3, bias=False)
            self.conv.init_weight_highpass(kernel)
            print("HP-3 initialized")
        elif kernel_size == 5:
            # 5x5 high pass
            kernel = np.array([[-1, -1, -1, -1, -1],
                               [-1, 1, 2, 1, -1],
                               [-1, 2, 4, 2, -1],
                               [-1, 1, 2, 1, -1],
                               [-1, -1, -1, -1, -1]], dtype=np.float32)
            self.conv = HighPassHard(in_channels=3, out_channels=3, kernel_size=5, stride=1, padding=5 // 2, dilation=1, groups=3, bias=False)
            self.conv.init_weight_highpass(kernel)
            print("HP-5 initialized")
        else:
            raise NotImplementedError()

    def forward(self, inputs):
        # Y←0.299⋅R+0.587⋅G+0.114⋅B
        # bgr to gray
        input_gray = 0.299 * inputs[:,2:3] + 0.587 * inputs[:,1:2] + 0.114 * inputs[:,0:1]
        # rgb to gray
        # input_gray = 0.299 * inputs[:,0:1] + 0.587 * inputs[:,1:2] + 0.114 * inputs[:,2:3]
        input_gray = torch.cat([input_gray]*3, dim=1)
        with torch.no_grad():
            if self.kernel_size == 2:
                f_h = self.conv_w(input_gray)
                f_w = self.conv_h(input_gray)
                f_out = (f_h + f_w) / 2.0
            else:
                f_out = self.conv(input_gray)
        return f_out


class HighPassHard(nn.Conv2d):
    """
    https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv2d
    normalize conv2d weight as a high pass filter form:
    weight[:][0, 0] = -1
    normalize(weight[:][i, j] while i*j!=0)
    """
    # def __init__(self, in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=3 // 2, dilation=1, groups=3, bias=False):
        # super(HighPassHard, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def init_weight_highpass(self, kernel):
        kernel = torch.from_numpy(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(3, *[1] * (kernel.dim() - 1))

        self.weight.data = kernel

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        with torch.no_grad():
            result = F.conv2d(input, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
            return result

if __name__ == "__main__":
    ten = torch.rand((1,3,4,4))
    hp3 = highPassEnsemble(kernel_size=5)
    ten_hp3 = hp3(ten)
    print(ten, ten_hp3)
    import pdb; pdb.set_trace()