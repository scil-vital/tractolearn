# -*- coding: utf-8 -*-

from torch import nn
from torch.nn import functional as F


class IncrFeatStridedConvFCUpsampReflectPadAE(nn.Module):
    """Strided convolution-upsampling-based AE using reflection-padding and
    increasing feature maps in decoder.
    """

    def __init__(self, latent_space_dims):
        super(IncrFeatStridedConvFCUpsampReflectPadAE, self).__init__()

        self.kernel_size = 3
        self.latent_space_dims = latent_space_dims

        self.pad = nn.ReflectionPad1d(1)

        def pre_pad(m):
            return nn.Sequential(self.pad, m)

        self.encod_conv1 = pre_pad(
            nn.Conv1d(3, 32, self.kernel_size, stride=2, padding=0)
        )
        self.encod_conv2 = pre_pad(
            nn.Conv1d(32, 64, self.kernel_size, stride=2, padding=0)
        )
        self.encod_conv3 = pre_pad(
            nn.Conv1d(64, 128, self.kernel_size, stride=2, padding=0)
        )
        self.encod_conv4 = pre_pad(
            nn.Conv1d(128, 256, self.kernel_size, stride=2, padding=0)
        )
        self.encod_conv5 = pre_pad(
            nn.Conv1d(256, 512, self.kernel_size, stride=2, padding=0)
        )
        self.encod_conv6 = pre_pad(
            nn.Conv1d(512, 1024, self.kernel_size, stride=1, padding=0)
        )

        self.fc1 = nn.Linear(8192, self.latent_space_dims)  # 8192 = 1024*8
        self.fc2 = nn.Linear(self.latent_space_dims, 8192)

        self.decod_conv1 = pre_pad(
            nn.Conv1d(1024, 512, self.kernel_size, stride=1, padding=0)
        )
        self.upsampl1 = nn.Upsample(
            scale_factor=2, mode="linear", align_corners=False
        )
        self.decod_conv2 = pre_pad(
            nn.Conv1d(512, 256, self.kernel_size, stride=1, padding=0)
        )
        self.upsampl2 = nn.Upsample(
            scale_factor=2, mode="linear", align_corners=False
        )
        self.decod_conv3 = pre_pad(
            nn.Conv1d(256, 128, self.kernel_size, stride=1, padding=0)
        )
        self.upsampl3 = nn.Upsample(
            scale_factor=2, mode="linear", align_corners=False
        )
        self.decod_conv4 = pre_pad(
            nn.Conv1d(128, 64, self.kernel_size, stride=1, padding=0)
        )
        self.upsampl4 = nn.Upsample(
            scale_factor=2, mode="linear", align_corners=False
        )
        self.decod_conv5 = pre_pad(
            nn.Conv1d(64, 32, self.kernel_size, stride=1, padding=0)
        )
        self.upsampl5 = nn.Upsample(
            scale_factor=2, mode="linear", align_corners=False
        )
        self.decod_conv6 = pre_pad(
            nn.Conv1d(32, 3, self.kernel_size, stride=1, padding=0)
        )

    def encode(self, x):
        h1 = F.relu(self.encod_conv1(x))
        h2 = F.relu(self.encod_conv2(h1))
        h3 = F.relu(self.encod_conv3(h2))
        h4 = F.relu(self.encod_conv4(h3))
        h5 = F.relu(self.encod_conv5(h4))
        h6 = self.encod_conv6(h5)

        self.encoder_out_size = (h6.shape[1], h6.shape[2])

        # Flatten
        h7 = h6.view(-1, self.encoder_out_size[0] * self.encoder_out_size[1])

        fc1 = self.fc1(h7)

        return fc1

    def decode(self, z):
        fc = self.fc2(z)
        fc_reshape = fc.view(
            -1, self.encoder_out_size[0], self.encoder_out_size[1]
        )
        h1 = F.relu(self.decod_conv1(fc_reshape))
        h2 = self.upsampl1(h1)
        h3 = F.relu(self.decod_conv2(h2))
        h4 = self.upsampl2(h3)
        h5 = F.relu(self.decod_conv3(h4))
        h6 = self.upsampl3(h5)
        h7 = F.relu(self.decod_conv4(h6))
        h8 = self.upsampl4(h7)
        h9 = F.relu(self.decod_conv5(h8))
        h10 = self.upsampl5(h9)
        h11 = self.decod_conv6(h10)

        return h11

    def forward(self, x):
        encoded = self.encode(x)
        return self.decode(encoded)
