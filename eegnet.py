from braindecode.models import EEGNet
import torch

def build_eegnet(n_channels=64, n_timepoints=321, n_classes=2, dropout=0.5):
    model = EEGNet(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_timepoints,
        drop_prob=dropout,
    )
    return model


if __name__ == "__main__":
    model = build_eegnet()
    x = torch.randn(8, 64, 321)   # (batch, channels, time)
    out = model(x)
    print("Output shape:", out.shape)  # expect (8, 2)