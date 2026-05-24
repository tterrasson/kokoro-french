import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, Conv2d
from torch.nn.utils.parametrizations import spectral_norm, weight_norm

from .utils import get_padding

LRELU_SLOPE = 0.1


def stft(x, fft_size, hop_size, win_length, window, log_mag=False):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    x_stft = torch.stft(x, fft_size, hop_size, win_length, window, return_complex=True)
    mag = torch.abs(x_stft)
    if log_mag:
        mag = torch.log1p(mag)

    return mag.transpose(2, 1)


class SpecDiscriminator(nn.Module):
    """docstring for Discriminator."""

    def __init__(
        self,
        fft_size=1024,
        shift_size=120,
        win_length=600,
        window="hann_window",
        use_spectral_norm=False,
        log_mag=False,
    ):
        super(SpecDiscriminator, self).__init__()
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.log_mag = log_mag
        self.register_buffer(
            "window",
            getattr(torch, window)(win_length),
            persistent=False,
        )
        self.discriminators = nn.ModuleList(
            [
                norm_f(nn.Conv2d(1, 32, kernel_size=(3, 9), padding=(1, 4))),
                norm_f(
                    nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))
                ),
                norm_f(
                    nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))
                ),
                norm_f(
                    nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))
                ),
                norm_f(
                    nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                ),
            ]
        )

        self.out = norm_f(nn.Conv2d(32, 1, 3, 1, 1))

    def forward(self, y):

        fmap = []
        y = y.squeeze(1)
        y = stft(
            y,
            self.fft_size,
            self.shift_size,
            self.win_length,
            self.window,
            log_mag=self.log_mag,
        )
        y = y.unsqueeze(1)

        for _, d in enumerate(self.discriminators):
            y = d(y)
            y = F.leaky_relu(y, LRELU_SLOPE)
            fmap.append(y)

        y = self.out(y)
        fmap.append(y)

        return torch.flatten(y, 1, -1), fmap


class MultiResSpecDiscriminator(torch.nn.Module):
    def __init__(
        self,
        fft_sizes=[1024, 2048, 512],
        hop_sizes=[120, 240, 50],
        win_lengths=[600, 1200, 240],
        window="hann_window",
        log_mag=False,
    ):

        super(MultiResSpecDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList(
            [
                SpecDiscriminator(
                    fft_size=fft_sizes[0],
                    shift_size=hop_sizes[0],
                    win_length=win_lengths[0],
                    window=window,
                    log_mag=log_mag,
                ),
                SpecDiscriminator(
                    fft_size=fft_sizes[1],
                    shift_size=hop_sizes[1],
                    win_length=win_lengths[1],
                    window=window,
                    log_mag=log_mag,
                ),
                SpecDiscriminator(
                    fft_size=fft_sizes[2],
                    shift_size=hop_sizes[2],
                    win_length=win_lengths[2],
                    window=window,
                    log_mag=log_mag,
                ),
            ]
        )

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for _, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super().__init__()
        self.period = period
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        1,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, periods=(2, 3, 5, 7, 11, 17)):
        super().__init__()
        self.periods = tuple(periods)
        self.discriminators = nn.ModuleList(
            [DiscriminatorP(period) for period in self.periods]
        )

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for _, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class STFTSubDiscriminator(nn.Module):
    def __init__(
        self,
        fft_size=1024,
        hop_size=256,
        win_length=1024,
        channels=32,
        log_mag=True,
        use_spectral_norm=False,
    ):
        super().__init__()
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.log_mag = log_mag
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)
        self.convs = nn.ModuleList(
            [
                norm_f(nn.Conv2d(1, channels, kernel_size=3, padding=1)),
                norm_f(
                    nn.Conv2d(
                        channels, channels * 2, kernel_size=3, stride=(1, 2), padding=1
                    )
                ),
                norm_f(
                    nn.Conv2d(
                        channels * 2,
                        channels * 4,
                        kernel_size=3,
                        stride=(2, 2),
                        padding=1,
                    )
                ),
                norm_f(nn.Conv2d(channels * 4, channels * 4, kernel_size=3, padding=1)),
            ]
        )
        self.post = norm_f(nn.Conv2d(channels * 4, 1, kernel_size=3, padding=1))

    def forward(self, x):
        fmap = []
        x = x.squeeze(1)
        x = stft(
            x,
            self.fft_size,
            self.hop_size,
            self.win_length,
            self.window,
            log_mag=self.log_mag,
        )
        x = x.unsqueeze(1)

        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)

        x = self.post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class MultiScaleSTFTDiscriminator(nn.Module):
    def __init__(
        self,
        resolutions=((1024, 256, 1024), (2048, 512, 2048), (512, 128, 512)),
        channels=32,
        log_mag=True,
        use_spectral_norm=False,
    ):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                STFTSubDiscriminator(
                    fft_size=fft_size,
                    hop_size=hop_size,
                    win_length=win_length,
                    channels=channels,
                    log_mag=log_mag,
                    use_spectral_norm=use_spectral_norm,
                )
                for fft_size, hop_size, win_length in resolutions
            ]
        )

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class SubBandSpecDiscriminator(nn.Module):
    def __init__(
        self,
        fft_size=1024,
        hop_size=256,
        win_length=1024,
        bands=((0, 64), (64, 192), (192, 384), (384, None)),
        channels=16,
        log_mag=True,
        use_spectral_norm=False,
    ):
        super().__init__()
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.bands = tuple((start, end) for start, end in bands)
        self.log_mag = log_mag
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)
        self.band_discriminators = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        norm_f(nn.Conv2d(1, channels, kernel_size=3, padding=1)),
                        norm_f(
                            nn.Conv2d(
                                channels,
                                channels * 2,
                                kernel_size=3,
                                stride=(1, 2),
                                padding=1,
                            )
                        ),
                        norm_f(
                            nn.Conv2d(
                                channels * 2,
                                channels * 4,
                                kernel_size=3,
                                stride=(2, 2),
                                padding=1,
                            )
                        ),
                        norm_f(nn.Conv2d(channels * 4, 1, kernel_size=3, padding=1)),
                    ]
                )
                for _ in self.bands
            ]
        )

    def forward_single_band(self, x, layers):
        fmap = []
        for idx, layer in enumerate(layers):
            x = layer(x)
            if idx < len(layers) - 1:
                x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        return torch.flatten(x, 1, -1), fmap

    def forward(self, x):
        x = x.squeeze(1)
        spec = stft(
            x,
            self.fft_size,
            self.hop_size,
            self.win_length,
            self.window,
            log_mag=self.log_mag,
        ).unsqueeze(1)

        outs, fmaps = [], []
        for (start, end), layers in zip(self.bands, self.band_discriminators):
            end = spec.shape[-1] if end is None else end
            out, fmap = self.forward_single_band(spec[..., start:end], layers)
            outs.append(out)
            fmaps.append(fmap)
        return outs, fmaps


class MultiSubBandSpecDiscriminator(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.discriminator = SubBandSpecDiscriminator(**kwargs)

    def forward(self, y, y_hat):
        y_d_rs, fmap_rs = self.discriminator(y)
        y_d_gs, fmap_gs = self.discriminator(y_hat)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class WavLMDiscriminator(nn.Module):
    """docstring for Discriminator."""

    def __init__(
        self,
        slm_hidden=768,
        slm_layers=13,
        initial_channel=64,
        use_spectral_norm=False,
        return_fmaps=False,
        dropout_p=0.05,
        use_group_norm=False,
    ):
        super(WavLMDiscriminator, self).__init__()
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.return_fmaps = return_fmaps
        self.dropout_p = dropout_p
        self.pre = norm_f(
            Conv1d(slm_hidden * slm_layers, initial_channel, 1, 1, padding=0)
        )
        self.pre_norm = nn.GroupNorm(1, initial_channel) if use_group_norm else nn.Identity()

        self.convs = nn.ModuleList(
            [
                norm_f(
                    nn.Conv1d(
                        initial_channel, initial_channel * 2, kernel_size=5, padding=2
                    )
                ),
                norm_f(
                    nn.Conv1d(
                        initial_channel * 2,
                        initial_channel * 4,
                        kernel_size=5,
                        padding=2,
                    )
                ),
                norm_f(
                    nn.Conv1d(initial_channel * 4, initial_channel * 4, 5, 1, padding=2)
                ),
            ]
        )

        self.conv_post = norm_f(Conv1d(initial_channel * 4, 1, 3, 1, padding=1))

    def forward(self, x):
        x = self.pre(x)
        x = self.pre_norm(x)
        x = F.leaky_relu(x, LRELU_SLOPE)

        fmap = []
        for conv in self.convs:
            x = F.dropout(x, p=self.dropout_p, training=self.training)
            x = conv(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)

        x = self.conv_post(x)
        x = torch.flatten(x, 1, -1)

        if self.return_fmaps:
            return x, fmap

        return x
