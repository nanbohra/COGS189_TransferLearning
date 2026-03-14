import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from tqdm import tqdm


### This implementation of EEGNet was provided by Srini to proceed with the project without braindecode dependencies.
class EEGNet_base(nn.Module):
    '''
    The other EEGNet doesn't have the depthwise-conv in the 2nd layer; and has biases in the conv layers.
    This one is the original EEGNet architecture.
    '''
    def __init__(self, PARAMS):
        super(EEGNet_base, self).__init__()

        self.srate = PARAMS["sampling_rate"]
        self.T = PARAMS["num_timepoints"]
        self.C = len(PARAMS["electrodes"])
        self.N = PARAMS["num_outputs"] 
        self.F1 = PARAMS["num_temporal_filters"]   # number of temporal filters
        self.D = PARAMS["num_spatial_filters"]     # depth, number of spatial filters
        self.F2 = self.D * self.F1

        # Block 1
        self.conv1 = nn.Conv2d(1, self.F1, (1, self.srate // 2), padding='same', bias=False)
        self.batchnorm1 = nn.BatchNorm2d(self.F1, False)
        self.conv2 = nn.Conv2d(self.F1, self.D * self.F1, (self.C, 1), padding='valid', groups=self.F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(self.D * self.F1, False)
        self.activation1 = nn.ELU()
        self.pooling1 = nn.AvgPool2d((1, 4))     # Makes timepoints 4 times smaller
        self.dropout1 = nn.Dropout(PARAMS["dropout"])

        # Block 2
        # Separable convolution, depthwise in time + pointwise
        self.conv3 = nn.Sequential(nn.Conv2d(self.F2, self.F2, (1, self.T // 8), padding='same', groups=self.F2, bias=False),
                                   nn.Conv2d(self.F2, self.F2, (1, 1), padding='same', bias=False))
        self.batchnorm3 = nn.BatchNorm2d(self.F2, False)
        self.activation2 = nn.ELU()
        self.pooling2 = nn.AvgPool2d((1, 5))
        self.dropout2 = nn.Dropout(PARAMS["dropout"])

        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in data.
        self.fc1 = nn.Linear(self.F2 * self.T // 20, self.N)
        
    def forward(self, x):
        # x is of shape (bs x 1 x C x T)
        bs = x.shape[0]

        # pdb.set_trace()
        # Block 1
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.activation1(x)
        x = self.pooling1(x)
        x = self.dropout1(x)
        
        # Block 2
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.activation2(x)
        x = self.pooling2(x)
        x = self.dropout2(x)
        
        # pdb.set_trace()
        
        # FC Layer
        x = x.reshape(bs, -1)
        
        outputs = self.fc1(x)
        
        return outputs



def build_eegnet(n_channels=64, n_timepoints=321, n_classes=2, dropout=0.5):
    # model = EEGNet(
    #     n_chans=n_channels,
    #     n_outputs=n_classes,
    #     n_times=n_timepoints,
    #     drop_prob=dropout,
    # )

    PARAMS = {
        "sampling_rate": 160,
        "num_timepoints": n_timepoints,
        "electrodes": list(range(n_channels)),
        "num_outputs": n_classes,
        "num_temporal_filters": 8, 
        "num_spatial_filters": 2,
        "dropout": dropout
    }
    return EEGNet_base(PARAMS)


if __name__ == "__main__":
    model = build_eegnet()
    x = torch.randn(8, 1, 64, 321)   # (batch,1,  channels, time)
    out = model(x)
    print("Output shape:", out.shape)  # expect (8, 2)