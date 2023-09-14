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

import torch
import math
import numpy as np
def sigmoid_schedule(t, start=-3, end=3, tau=1, clip_min=1e-9):
    def sigmoid(x):
        return 1 / (1 + torch.exp(-x))
    def scalar_sigmoid(x):
        return 1 / (1 + np.exp(-x))
    v_start = scalar_sigmoid(start / tau)
    v_end = scalar_sigmoid(end / tau)
    output = sigmoid((t * (end-start) + start) / tau)
    output = (v_end - output) / (v_end - v_start)
    return torch.clamp(output, clip_min, 1.)

def cosine_schedule(t, start=0, end=1, tau=1, clip_min=1e-9):
    v_start = math.cos(start * math.pi / 2) ** (2 * tau)
    v_end = math.cos(end * math.pi / 2) ** (2 * tau)
    output = math.cos((t * (end - start) + start) * math.pi / 2) ** (2 * tau)
    output = (v_end - output) / (v_end - v_start)
    return np.clip(output, clip_min, 1.)

def enforce_zero_terminal_snr(betas):
    alphas = 1 - betas
    alphas_bar = alphas.cumprod(dim=0)
    alphas_bar_sqrt = alphas_bar.sqrt()

    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone() - 1e-4

    alphas_bar_sqrt -= alphas_bar_sqrt_T
    alphas_bar_sqrt *= (alphas_bar_sqrt_0) / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)
    alphas_bar = alphas_bar_sqrt ** 2
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1- alphas
    return betas


import matplotlib.pyplot as plt

# steps = torch.linspace(float(0.000001), float(1.), int(51))
# alphas = sigmoid_schedule(steps, tau=0.5)
# alphas = alphas.float().cpu()

# # steps = np.linspace(float(0.000001), float(1.), int(51))
# # alphas = torch.Tensor([cosine_schedule(t, start=0.0, tau=.4) for t in steps])
# # alphas = alphas.float().cpu()

# # alphas *= 0.05
# # alphas += 0.95


betas = torch.linspace(1e-4, 0.05, 50)
betas = enforce_zero_terminal_snr(betas)

# # plt.plot(alphas.numpy(), label='alpha_true')

# predefined_alpha = (alphas[1:] / alphas[:-1])
# predefined_alpha[0] *= alphas[0]
# betas = 1 - predefined_alpha

# betas = betas.clamp(max=0.999)
# # alphas = alphas.clamp(min=0.00001, max=0.9999)
# noise_schedule = betas.float()
# # diffusion_hyperparams = compute_hyperparams_given_schedule(noise_schedule)

plt.plot(betas.detach().cpu().numpy(), label='beta_true')
# plt.plot(alphas.detach().cpu().numpy(), label='alphas_true')
plt.legend()
plt.savefig('noise_infer_alphas.png')

params = AttrDict(
    # Training params
    batch_size=16,
    learning_rate=2e-4,
    max_grad_norm=None,
    use_l2loss=True,
    audio_channels=2,

    # Data params
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
    condition_prior=False, # whether to use energy prior as concatenated feature with mel. default is false
    condition_prior_global=False, # whether to use energy prior as global condition with projection. default is false
    # minimum std that clips the prior std below std_min. ensures numerically stable training.
    std_min=0.1,
    # whether to clip max energy to certain value. Affects normalization of the energy.
    # Lower value -> more data points assign to ~1 variance. so pushes latent space to higher variance regime
    # if None, no override, uses computed stat
    # for volume-normalized waveform with HiFi-GAN STFT, max energy of 4 gives reasonable range that clips outliers
    max_energy_override=4.,

    # Model params
    residual_layers=30,
    residual_channels=32,
    dilation_cycle_length=7,
    noise_schedule=betas.numpy().tolist(),
    # inference_noise_schedule=noise_schedule.numpy().tolist(),
    # noise_schedule=np.linspace(1e-4, 0.05, 50).tolist(), # [beta_start, beta_end, num_diffusion_step]
    # inference_noise_schedule=np.linspace(1e-4, 0.05, 50).tolist(), # [beta_start, beta_end, num_diffusion_step]
    inference_noise_schedule=[0.0001, 0.001, 0.01, 0.05, 0.2, 0.5], # T>=50
    # inference_noise_schedule=[0.001, 0.01, 0.05, 0.2] # designed for for T=20
)
