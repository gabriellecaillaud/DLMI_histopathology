import torch.nn as nn
import torch.nn.functional as F
    
class LoraConv(nn.Module):
    def __init__(self,conv_module: nn.Conv2d, r=4,alpha=4):
        """
        We create a LoRa convolution that will be the sum of the original convolution
        and the LoRa convoltion
        """
        super().__init__()
        self.alpha=alpha
        self.r=r
        
        self.convA=nn.Conv2d(in_channels=conv_module.in_channels, 
                             out_channels=r,
                             kernel_size=conv_module.kernel_size,
                             padding=conv_module.padding,
                             stride=conv_module.stride,bias=False)
        
        self.convB=nn.Conv2d(in_channels=r,
                             out_channels=conv_module.out_channels,
                             kernel_size=1,
                             padding=0,
                             stride=1,bias=False)

    
    def forward(self, x):
        """ 
        We now forward pass using only the LoRa weights from AB
        """
        other_x=self.convA(x)
        other_x=self.convB(other_x)
        return self.alpha/self.r * other_x


    
class LoraLinear(nn.Module):
    def __init__(self,lin_module, r= 4, alpha=4) -> None:
        super().__init__()
        in_features=lin_module.in_features
        out_features=lin_module.out_features
        self.lora_A = nn.Linear(in_features=in_features,out_features=r)
        self.lora_B = nn.Linear(in_features=r,out_features=out_features)
        self.r=r
        self.alpha=alpha
        
    def forward(self, x):
        result = self.lora_B(self.lora_A(x))
        return self.alpha/self.r * result
