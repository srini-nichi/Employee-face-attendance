# Import necessary PyTorch modules and layers
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
from torch import nn
import torch
import math

# Custom Flatten layer to convert multi-dimensional input to 2D (batch, features)
class Flatten(Module):
    def forward(self, input):
        # Flattens the input tensor to (batch_size, features)
        return input.view(input.size(0), -1)

# Function to perform L2 normalization on the input tensor
def l2_norm(input, axis=1):
    # Compute the L2 norm along the specified axis
    norm = torch.norm(input, 2, axis, True)
    # Divide the input by the computed norm to normalize the data
    output = torch.div(input, norm)
    return output

##################################  MobileFaceNet #############################################################
# Feature Extraction (MobileFaceNet): Extracts meaningful face embeddings from the input images.

# Conv_block: A basic convolution block with Conv2D, BatchNorm, and PReLU activation.
class Conv_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)
        self.prelu = PReLU(out_c)  # Parametric ReLU activation
    def forward(self, x):
        # Forward pass through conv -> batchnorm -> activation
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x

# Linear_block: Similar to Conv_block, but without activation
class Linear_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)
    def forward(self, x):
        # Forward pass through conv -> batchnorm (no activation here)
        x = self.conv(x)
        x = self.bn(x)
        return x

# Depth_Wise: Implements depthwise separable convolution with optional residual connection
class Depth_Wise(Module):
    def __init__(self, in_c, out_c, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(Depth_Wise, self).__init__()
        # First pointwise (1x1) conv block
        self.conv = Conv_block(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        # Depthwise (grouped) convolution
        self.conv_dw = Conv_block(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride)
        # Project back to required number of channels
        self.project = Linear_block(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual  # If residual is True, a skip connection will be added
    def forward(self, x):
        if self.residual:
            short_cut = x  # Save the input for the residual connection
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x  # Add input back to output (residual connection)
        else:
            output = x
        return output

# Residual: Implements a series of Depth_Wise blocks with residual connections
class Residual(Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        modules = []
        # Stack multiple Depth_Wise blocks to form a residual block
        for _ in range(num_block):
            modules.append(Depth_Wise(c, c, residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups))
        self.model = Sequential(*modules)  # Combine them in sequence
    def forward(self, x):
        return self.model(x)

# MobileFaceNet: The main MobileFaceNet model for face embedding generation
class MobileFaceNet(Module):
    def __init__(self, embedding_size):
        super(MobileFaceNet, self).__init__()
        # Series of convolution and residual blocks for feature extraction
        self.conv1 = Conv_block(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))  # Initial convolution
        self.conv2_dw = Conv_block(64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)  # Depthwise conv
        self.conv_23 = Depth_Wise(64, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)  # Depthwise separable conv
        self.conv_3 = Residual(64, num_block=4, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1))  # Residual block
        self.conv_34 = Depth_Wise(64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        self.conv_4 = Residual(128, num_block=6, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_45 = Depth_Wise(128, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512)
        self.conv_5 = Residual(128, num_block=2, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_6_sep = Conv_block(128, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))  # Final 1x1 conv layer
        self.conv_6_dw = Linear_block(512, 512, groups=512, kernel=(7,7), stride=(1, 1), padding=(0, 0))  # Global average pooling conv
        self.conv_6_flatten = Flatten()  # Flatten the output for fully connected layer
        self.linear = Linear(512, embedding_size, bias=False)  # Final linear layer to generate embeddings
        self.bn = BatchNorm1d(embedding_size)  # Normalize embeddings
        
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))  # He initialization for conv layers
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)  # BatchNorm weight set to 1
                m.bias.data.zero_()  # BatchNorm bias set to 0
    
    def forward(self, x):
        # Forward pass through all the layers
        out = self.conv1(x)
        out = self.conv2_dw(out)
        out = self.conv_23(out)
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)
        out = self.conv_6_sep(out)
        out = self.conv_6_dw(out)
        out = self.conv_6_flatten(out)  # Flatten the feature map
        out = self.linear(out)  # Linear layer for embeddings
        out = self.bn(out)  # Normalize the embeddings
        return l2_norm(out)  # L2-normalize the embeddings

##################################  Arcface head #############################################################
# Classification (ArcFace): Compares these embeddings to known classes (faces) using an angular margin-based loss, ensuring high accuracy in distinguishing between similar faces.
class Arcface(Module):
    # Additive margin softmax loss as per https://arxiv.org/abs/1801.05599
    def __init__(self, embedding_size=512, classnum=51332, s=64., m=0.5):
        super(Arcface, self).__init__()
        self.classnum = classnum  # Number of classes
        self.kernel = Parameter(torch.Tensor(embedding_size, classnum))  # Weight matrix
        nn.init.xavier_uniform_(self.kernel)  # Xavier initialization
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.m = m  # Margin
        self.s = s  # Scaling factor
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embbedings, label):
        # Normalize weights and embeddings
        nB = len(embbedings)  # Batch size
        kernel_norm = l2_norm(self.kernel, axis=0)  # L2-normalize weights
        cos_theta = torch.mm(embbedings, kernel_norm)  # Cosine similarity between embeddings and class weights
        cos_theta = cos_theta.clamp(-1, 1)  # Clamp to valid cosine range [-1, 1]
        target_logit = cos_theta[torch.arange(0, nB), label].view(-1, 1)  # Logits for correct classes
        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))  # sin(theta)
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # Additive angular margin
        mask = cos_theta > self.th
        hard_example = cos_theta[mask]
        with torch.no_grad():
            cos_theta[mask] = hard_example * 1.0
        cos_theta.scatter_(1, label.view(-1, 1).long(), cos_theta_m)
        output = cos_theta * self.s  # Scale the logits
        return output
