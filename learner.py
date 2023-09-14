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
import os
import torch
import torch.nn as nn
from pathlib import Path

from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchaudio
from pytorch_wavelets import DWT1DForward, DWT1DInverse

from dataset import from_path as dataset_from_path
from dataset import from_path_valid as dataset_from_path_valid
from model import PriorGrad
from preprocess import get_mel
from stft_loss import MultiResolutionSTFTLoss

def _nested_map(struct, map_fn):
    if isinstance(struct, tuple):
        return tuple(_nested_map(x, map_fn) for x in struct)
    if isinstance(struct, list):
        return [_nested_map(x, map_fn) for x in struct]
    if isinstance(struct, dict):
        return {k: _nested_map(v, map_fn) for k, v in struct.items()}
    return map_fn(struct)

def scaled_mse_loss(decoder_output, target, target_std):
    # inverse of diagonal matrix is 1/x for each element
    sigma_inv = torch.reciprocal(target_std)
    mse_loss = (((decoder_output - target) * sigma_inv) ** 2)
    mse_loss = (mse_loss).sum() / torch.numel(decoder_output)
    return mse_loss

def remove_cutoff_frequency(signal):
    signal = torchaudio.functional.highpass_biquad(signal,
                                                   sample_rate=22050//2,
                                                   cutoff_freq=15)
    signal = torchaudio.functional.lowpass_biquad(signal,
                                                  sample_rate=22050/2,
                                                  cutoff_freq=5500)
    return signal

import random
def pyramid_noise_like(x, discount=0.9):
    b, c, w = x.shape # EDIT: w and h get over-written, rename for a different variant!
    device = x.device
    u = nn.Upsample(size=(w))
    noise = torch.randn_like(x).to(device)
    for i in range(7):
        r = random.random()*2+2 # Rather than always going 2x, 
        w = max(1, int(w/(r**i))) #, max(1, int(h/(r**i)))
        noise += u(torch.rand_like(x).to(device)) * discount**i
        if w==1: break # Lowest resolution is 1x1
    return noise/noise.std() # Scaled back to roughly unit variance


class PriorGradLearner:
    def __init__(self, model_dir, model, dataset, dataset_val, optimizer, params, *args, **kwargs):
        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir
        self.model = model
        self.dataset = dataset
        self.dataset_val = dataset_val
        self.optimizer = optimizer
        self.params = params
        self.autocast = torch.cuda.amp.autocast(enabled=kwargs.get('fp16', False))
        self.scaler = torch.cuda.amp.GradScaler(enabled=kwargs.get('fp16', False))
        self.step = 0
        self.is_master = True

        self.use_l2loss = params.use_l2loss

        self.use_prior = params.use_prior
        self.condition_prior = params.condition_prior
        self.condition_prior_global = params.condition_prior_global

        assert not (self.condition_prior and self.condition_prior_global),\
            "use only one of the following parameter: condition_prior or condition_prior_global"

        beta = np.array(self.params.noise_schedule)
        noise_level = np.cumprod(1 - beta)
        self.noise_level = torch.tensor(noise_level.astype(np.float32))

        self.summary_writer = None
        self.dwt = DWT1DForward().cuda()
        self.idwt = DWT1DInverse().cuda()
        self.stft_loss = MultiResolutionSTFTLoss(fft_sizes=[512, 1024, 256], hop_sizes=[50, 120, 30], win_lengths=[300, 600, 128]).cuda()

    def state_dict(self):
        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        return {
            'step': self.step,
            'model': {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model_state.items()},
            'optimizer': {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in
                          self.optimizer.state_dict().items()},
            'params': dict(self.params),
            'scaler': self.scaler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            self.model.module.load_state_dict(state_dict['model'])
        else:
            self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scaler.load_state_dict(state_dict['scaler'])
        self.step = state_dict['step']

    def save_to_checkpoint(self, filename='weights'):
        save_basename = f'{filename}-{self.step}.pt'
        save_name = f'{self.model_dir}/{save_basename}'
        link_name = f'{self.model_dir}/{filename}.pt'
        torch.save(self.state_dict(), save_name)
        if os.name == 'nt':
            torch.save(self.state_dict(), link_name)
        else:
            if os.path.islink(link_name):
                os.unlink(link_name)
            os.symlink(save_basename, link_name)

    def restore_from_checkpoint(self, filename='weights'):
        try:
            checkpoint = torch.load(f'{self.model_dir}/{filename}.pt')
            self.load_state_dict(checkpoint)
            return True
        except FileNotFoundError:
            return False

    def train(self, max_steps=None):
        device = next(self.model.parameters()).device
        while True:
            for features in tqdm(self.dataset,
                                 desc=f'Epoch {self.step // len(self.dataset)}') if self.is_master else self.dataset:
                if max_steps is not None and self.step > max_steps:
                    return
                features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
                loss, grad_loss, sc_loss, mag_loss_l, mag_loss_h, predicted = self.train_step(features)
                if torch.isnan(loss).any():
                    raise RuntimeError(f'Detected NaN loss at step {self.step}.')
                if self.is_master:
                    if self.step % 50 == 0:
                        self._write_summary(self.step, features, loss, grad_loss, sc_loss, mag_loss_l, mag_loss_h)
                    if self.step % 10000 == 0:
                        self.run_valid_loop()
                    if self.step % 50000 == 0:
                        print("INFO: saving checkpoint at step {}".format(self.step))
                        self.save_to_checkpoint()
                self.step += 1

    def train_step(self, features):
        for param in self.model.parameters():
            param.grad = None

        audio = features['audio']
        spectrogram = features['spectrogram']
        target_std_lb = features['target_std']
        target_std_hb = features['target_std_hb']

        if self.condition_prior:
            target_std_specdim = target_std[:, ::self.params.hop_samples].unsqueeze(1)
            spectrogram = torch.cat([spectrogram, target_std_specdim], dim=1)
            global_cond = None
        elif self.condition_prior_global:
            target_std_specdim = target_std[:, ::self.params.hop_samples].unsqueeze(1)
            global_cond = target_std_specdim
        else:
            global_cond = None


        N, T = audio.shape
        device = audio.device
        self.dwt = self.dwt.to(device)
        audio = audio[:, None, :]
        l, [h] = self.dwt(audio)
        audio = torch.cat((l, h), dim=1) # [N, 2, T//2]

        target_std = torch.cat((target_std_lb[:, None, :],
                                target_std_hb[:, None, :]), dim=1)
        
        self.noise_level = self.noise_level.to(device)

        with self.autocast:
            t = torch.randint(0, len(self.params.noise_schedule), [N], device=audio.device)
            noise_scale = self.noise_level[t].unsqueeze(1).unsqueeze(2)
            noise_scale_sqrt = noise_scale ** 0.5
            noise = torch.randn_like(audio) # + 0.01 * torch.randn(audio.shape[0], audio.shape[1], 1).to(audio.device)
            noise = noise * target_std
            noisy_audio = noise_scale_sqrt * audio + (1.0 - noise_scale) ** 0.5 * noise

            predicted = self.model(noisy_audio, spectrogram, t, global_cond)

            if self.use_prior:
                if self.use_l2loss:
                    predicted_l, predicted_h = torch.chunk(predicted, 2, dim=1)
                    loss_l = scaled_mse_loss(predicted_l.squeeze(1), noise[:, 0, :], target_std_lb)
                    loss_h = scaled_mse_loss(predicted_h.squeeze(1), noise[:, 1, :], target_std_hb)
                    grad_loss = loss_l + loss_h
                    sc_loss, mag_loss_l = self.stft_loss(noise[:, 0, :], predicted_l.squeeze(1))
                    sc_loss, mag_loss_h = self.stft_loss(noise[:, 1, :], predicted_h.squeeze(1))
                    loss = grad_loss + 1e-1 * (mag_loss_l + mag_loss_h)
                else:
                    raise NotImplementedError
            else:
                if self.use_l2loss:
                    loss = nn.MSELoss()(noise, predicted.squeeze(1))
                else:
                    loss = nn.L1Loss()(noise, predicted.squeeze(1))

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        self.grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.params.max_grad_norm or 1e9)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss, grad_loss, sc_loss, mag_loss_l, mag_loss_h, predicted

    def run_valid_loop(self):
        with torch.no_grad():
            device = next(self.model.parameters()).device

            losses = []
            losses_l1 = []
            audio_preds = []

            for features in tqdm(self.dataset_val,
                                 desc=f'Valid {len(self.dataset_val)}') if self.is_master else self.dataset_val:
                features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)

                audio = features['audio']
                spectrogram = features['spectrogram']
                target_std_lb = features['target_std']
                target_std_hb = features['target_std_hb']

                if self.condition_prior:
                    target_std_specdim = target_std[:, ::self.params.hop_samples].unsqueeze(1)
                    spectrogram = torch.cat([spectrogram, target_std_specdim], dim=1)
                    global_cond = None
                elif self.condition_prior_global:
                    target_std_specdim = target_std[:, ::self.params.hop_samples].unsqueeze(1)
                    global_cond = target_std_specdim
                else:
                    global_cond = None

                N, T = audio.shape
                device = audio.device
                self.dwt = self.dwt.to(device)
                l, [h] = self.dwt(audio[:, None, :])
                audio = torch.cat((l, h), dim=1) # [N, 2, T//2]

                target_std = torch.cat((target_std_lb[:, None, :],
                                        target_std_hb[:, None, :]), dim=1)
                
                self.noise_level = self.noise_level.to(device)

                t = torch.randint(0, len(self.params.noise_schedule), [N], device=audio.device)
                noise_scale = self.noise_level[t].unsqueeze(1).unsqueeze(2)
                noise_scale_sqrt = noise_scale ** 0.5
                noise = torch.randn_like(audio)
                noise = noise * target_std
                noisy_audio = noise_scale_sqrt * audio + (1.0 - noise_scale) ** 0.5 * noise

                if hasattr(self.model, 'module'):
                    predicted = self.model.module(noisy_audio, spectrogram, t, global_cond)
                else:
                    predicted = self.model(noisy_audio, spectrogram, t, global_cond)

                if self.use_prior:
                    if self.use_l2loss:
                        predicted_l, predicted_h = torch.chunk(predicted, 2, dim=1)
                        loss_l = scaled_mse_loss(predicted_l.squeeze(1), noise[:, 0, :], target_std_lb)
                        loss_h = scaled_mse_loss(predicted_h.squeeze(1), noise[:, 1, :], target_std_hb)
                        grad_loss = loss_l + loss_h
                        sc_loss, mag_loss_l = self.stft_loss(noise[:, 0, :], predicted_l.squeeze(1))
                        sc_loss, mag_loss_h = self.stft_loss(noise[:, 1, :], predicted_h.squeeze(1))
                        loss = grad_loss + 1e-1 * (mag_loss_l + mag_loss_h)
                    else:
                        raise NotImplementedError
                else:
                    if self.use_l2loss:
                        loss = nn.MSELoss()(noise, predicted.squeeze(1))
                    else:
                        loss = nn.L1Loss()(noise, predicted.squeeze(1))

                losses.append(loss.cpu().numpy())

                audio_pred = self.predict(spectrogram, target_std, global_cond)
                self.idwt = self.idwt.to(device=audio_pred.device)
                audio_pred = self.idwt((audio_pred[:, 0, :].unsqueeze(1), [audio_pred[:, 1, :].unsqueeze(1)])).squeeze(1)
                audio_pred = torch.clamp(audio_pred, -1.0, 1.0)
                audio_preds.append(audio_pred.cpu().numpy())

                loss_l1 = torch.nn.L1Loss()(get_mel(audio_pred.squeeze(0), self.params), spectrogram).item()
                losses_l1.append(loss_l1)

            loss_valid = np.mean(losses)
            loss_l1 = np.mean(losses_l1)
            self._write_summary_valid(self.step, loss_valid, loss_l1, audio_preds, grad_loss, sc_loss, mag_loss_l, mag_loss_h)

    def predict(self, spectrogram, target_std, global_cond=None):
        with torch.no_grad():
            device = next(self.model.parameters()).device
            # --------------------------------------
            # alpha -> talpha
            # beta -> training_noise_schedule
            # gamma -> alpha
            # eta -> beta
            training_noise_schedule = np.array(self.params.noise_schedule)
            inference_noise_schedule = np.array(self.params.inference_noise_schedule)

            talpha = 1 - training_noise_schedule
            talpha_cum = np.cumprod(talpha)

            beta = inference_noise_schedule
            alpha = 1 - beta
            alpha_cum = np.cumprod(alpha)

            T = []
            for s in range(len(inference_noise_schedule)):
                for t in range(len(training_noise_schedule) - 1):
                    if talpha_cum[t + 1] <= alpha_cum[s] <= talpha_cum[t]:
                        twiddle = (talpha_cum[t] ** 0.5 - alpha_cum[s] ** 0.5) / (
                                    talpha_cum[t] ** 0.5 - talpha_cum[t + 1] ** 0.5)
                        T.append(t + twiddle)
                        break
            T = np.array(T, dtype=np.float32)

            # Expand rank 2 tensors by adding a batch dimension.
            if len(spectrogram.shape) == 2:
                spectrogram = spectrogram.unsqueeze(0)
            spectrogram = spectrogram.to(device)

            audio = torch.randn(spectrogram.shape[0], self.params.audio_channels, self.params.hop_samples * spectrogram.shape[-1] // self.params.audio_channels,
                                device=device)
            audio = audio * target_std
            noise_scale = torch.from_numpy(alpha_cum ** 0.5).float().unsqueeze(1).to(device)

            for n in range(len(alpha) - 1, -1, -1):
                c1 = 1 / alpha[n] ** 0.5
                c2 = beta[n] / (1 - alpha_cum[n]) ** 0.5
                if hasattr(self.model, 'module'):
                    audio = c1 * (audio - c2 * self.model.module(audio, spectrogram, torch.tensor([T[n]], device=audio.device),
                                                                 global_cond).squeeze(1))
                else:
                    audio = c1 * (audio - c2 * self.model(audio, spectrogram, torch.tensor([T[n]], device=audio.device),
                                                          global_cond).squeeze(1))
                if n > 0:
                    noise = torch.randn_like(audio)
                    noise = noise * target_std
                    sigma = ((1.0 - alpha_cum[n - 1]) / (1.0 - alpha_cum[n]) * beta[n]) ** 0.5
                    audio += sigma * noise
                audio = torch.clamp(audio, -1.0, 1.0)

            return audio

    def _write_summary(self, step, features, loss, grad_loss, sc_loss, mag_loss_l, mag_loss_h):
        writer = self.summary_writer or SummaryWriter(self.model_dir, purge_step=step)
        writer.add_audio('feature/audio', features['audio'][0], step, sample_rate=self.params.sample_rate)
        writer.add_image('feature/spectrogram', torch.flip(features['spectrogram'][:1], [1]), step)
        writer.add_scalar('train/loss', loss, step)
        writer.add_scalar('train/loss_grad', grad_loss, step)
        writer.add_scalar('train/loss_sc', sc_loss, step)
        writer.add_scalar('train/loss_mag_h', mag_loss_h, step)
        writer.add_scalar('train/loss_mag_l', mag_loss_l, step)
        writer.add_scalar('train/grad_norm', self.grad_norm, step)
        writer.flush()
        self.summary_writer = writer

    def _write_summary_valid(self, step, loss, loss_l1, audio_preds, grad_loss, sc_loss, mag_loss_l, mag_loss_h):
        writer = self.summary_writer or SummaryWriter(self.model_dir, purge_step=step)
        for i in range(len(audio_preds)):
            writer.add_audio('valid/audio_pred_{}'.format(i), audio_preds[i], step, sample_rate=self.params.sample_rate)
        writer.add_scalar('valid/loss', loss, step)
        writer.add_scalar('valid/loss_lsmae', loss_l1, step)
        writer.add_scalar('valid/loss_grad', grad_loss, step)
        writer.add_scalar('valid/loss_sc', sc_loss, step)
        writer.add_scalar('valid/loss_mag_l', mag_loss_l, step)
        writer.add_scalar('valid/loss_mag_l', mag_loss_h, step)
        writer.flush()
        self.summary_writer = writer


def _train_impl(replica_id, model, dataset, dataset_val, args, params):
    torch.backends.cudnn.benchmark = True
    opt = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

    learner = PriorGradLearner(args.model_dir, model, dataset, dataset_val, opt, params, fp16=args.fp16)
    learner.is_master = (replica_id == 0)
    learner.restore_from_checkpoint()
    learner.train(max_steps=args.max_steps)


def train(args, params):
    dataset = dataset_from_path(args.data_root, args.filelist, params)
    dataset_val = dataset_from_path_valid(args.data_root, os.path.join(Path(args.filelist).parent, "valid.txt"), params)
    model = PriorGrad(params).cuda()
    _train_impl(0, model, dataset, dataset_val, args, params)


def train_distributed(replica_id, replica_count, port, args, params):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    torch.distributed.init_process_group('nccl', rank=replica_id, world_size=replica_count)

    device = torch.device('cuda', replica_id)
    torch.cuda.set_device(device)
    model = PriorGrad(params).to(device)
    model = DistributedDataParallel(model, device_ids=[replica_id])
    dataset = dataset_from_path(args.data_root, args.filelist, params, is_distributed=True)
    if replica_id == 0:
        dataset_val = dataset_from_path_valid(args.data_root, os.path.join(Path(args.filelist).parent, "valid.txt"), params, is_distributed=False)
    else:
        dataset_val = None
    _train_impl(replica_id, model, dataset, dataset_val, args, params)
