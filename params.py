# Copyright 2022 (c) Microsoft Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import torch
import math

import matplotlib.pyplot as plt


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def override(self, attrs):
        if isinstance(attrs, dict):
            self.__dict__.update(**attrs)
        elif isinstance(attrs, (list, tuple, set)):
            for attr in attrs:
                self.override(attr)
        elif attrs is not None:
            raise NotImplementedError
        return self


def enforce_zero_terminal_snr(gammas):
    """Shift the noise schedule in order to achieve the zero-SNR at very first step
    Please refer to https://arxiv.org/abs/2305.08891

    This code corresponds to Equation 8, and Fig.4 in our paper

    Args:
        gammas (torch.Tensor): Into gamma function

    Returns:
        torch.Tensor: zero-snr noise scheduler
    """
    alphas = 1 - gammas
    alphas_bar = alphas.cumprod(dim=0)
    alphas_bar_sqrt = alphas_bar.sqrt()

    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone() - 1e-4

    alphas_bar_sqrt -= alphas_bar_sqrt_T
    alphas_bar_sqrt *= (alphas_bar_sqrt_0) / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)
    alphas_bar = alphas_bar_sqrt**2
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    alphas = torch.cat([alphas_bar[0:1], alphas])
    gammas = 1 - alphas
    return gammas


# * Predefined gamma function as shown in Fig.4
gammas = torch.linspace(1e-4, 0.05, 50)
gammas = enforce_zero_terminal_snr(gammas)

params = AttrDict(
    # Training params
    batch_size=16,
    learning_rate=2e-4,
    max_grad_norm=None,
    use_l2loss=True,
    audio_channels=2,
    # * Data params, Sec. 4.1
    sample_rate=22050,
    n_mels=80,
    n_fft=1024,
    hop_samples=256,
    fmin=0,
    fmax=8000,
    crop_mel_frames=62,  # PriorGrad keeps the previous open-source implementation
    # new data params for PriorGrad-vocoder
    use_prior=True,
    # optional parameters to additionally use the frame-level energy as the conditional input
    # one can choose one of the two options as below. note that only one can be set to True.
    condition_prior=False,  # whether to use energy prior as concatenated feature with mel. default is false
    condition_prior_global=False,  # whether to use energy prior as global condition with projection. default is false
    # minimum std that clips the prior std below std_min. ensures numerically stable training.
    std_min=0.1,
    # whether to clip max energy to certain value. Affects normalization of the energy.
    # Lower value -> more data points assign to ~1 variance. so pushes latent space to higher variance regime
    # if None, no override, uses computed stat
    # for volume-normalized waveform with HiFi-GAN STFT, max energy of 4 gives reasonable range that clips outliers
    max_energy_override=4.0,
    # Model params
    # * Here we change the the paprameter as shown in Sec 4.1
    residual_layers=30,
    residual_channels=32,
    dilation_cycle_length=7,
    noise_schedule=gammas.numpy().tolist(),  # We use zero-SNR noise schedule to train model and inference process
    inference_noise_schedule=gammas.numpy().tolist(),  # T=50
    # Inference only
    enable_remove_cutoff_alias=True,
)
