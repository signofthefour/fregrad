## Official implementation of  __FreGrad: Lightweight and fast frequency-aware diffusion vocoder (ICASSP 2024)__

This repository is an official PyTorch implementation of the paper:

> [Tan Dat Nguyen](https://signofthefour.github.io/)*, [Ji-Hoon Kim](https://sites.google.com/view/jhoonkim/)*, [Youngjoon Jang](https://art-jang.github.io/), Jaehun Kim, [Joon Son Chung](https://mmai.io/joon/). "FreGrad: lightweight and fast frequency-aware diffusion vocoder." _ICASSP_ (2024).
>[[arxiv]](https://arxiv.org/abs/2401.10032)
>[[demo]](https://mm.kaist.ac.kr/projects/FreGrad)
>[[MMAI-KAIST]](https://mm.kaist.ac.kr/)

![](./pics/fregrad.gif)

This repository contains a vocoder model (mel-spectrogram conditional waveform synthesis) presented in FreGrad.

## Abstract
The goal of this paper is to generate realistic audio with a lightweight and fast diffusion-based vocoder, named FreGrad. Our framework consists of the following three key components: (1) We employ discrete wavelet transform that decomposes a complicated waveform into sub-band wavelets, which helps FreGrad to operate on a simple and concise feature space, (2) We design a frequency-aware dilated convolution that elevates frequency awareness, resulting in generating speech with accurate frequency information, and (3) We introduce a bag of tricks that boosts the generation quality of the proposed model. In our experiments, FreGrad achieves $3.7$ times faster training time and $2.2$ times faster inference speed compared to our baseline while reducing the model size by $0.6$ times (only $1.78$ M parameters) without sacrificing the output quality.

## Demo

Refer to the [demo page](https://mm.kaist.ac.kr/projects/FreGrad) for the samples from the model.

## Quick Start and Examples

> I recommend user to use VSCode Better Comments to easily find out our comments that show our contributions as described in paper.

1. Navigate to FreGrad root and install dependencies
   ```bash
   # the codebase has been tested on Python 3.8 with PyTorch 1.8.2 LTS and 1.10.2 conda binaries
   pip install -r requirements.txt
   chmod +x train.sh inference.sh
   ```

2. Modify `filelists/train.txt`, `filelists/valid.txt`, `filelists/test.txt` so that the filelists point to the absolute path of the wav files. The codebase provides the LJSpeech dataset template. Here, we also provided randomly generated filelists we used to train our model that reported in paper.

3. Train FreGrad (our training code supports multi-GPU training). To train the model:
   -  Take a look and change default parameters defined in params.py if needed.
   - Specify cuda devices before train.

   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3 ./train.sh
   ```
   The training script first builds the training set statistics and saves it to `stats_priorgrad` folder created at `data_root` (`/path/to/your/LJSpeech-1.1` in the above example).

   It also automatically saves the hyperparameter file (`params.py`), renamed as `params_saved.py`, to `model_dir` at runtime to be used for inference.

4. Inference (fast mode with T=6)
   ```bash
   CUDA_VISIBLE_DEVICES=0 ./inference.sh
   ```
   Please uncomment or comment options in `inference.sh` to control inference process. Here, we provided:
      - `--fast` `--fast_iter 6` uses fast inference noise schedule with `--fast-iter` reverse diffusion steps.
   
      - If `--fast` is not provided, the model performs slow sampling with the same `T` step forward diffusion used in training.

   Samples are saved to the `sample_fast` if `--fast` is used, or `sample_slow` if not, created at the parent directory of the model (`checkpoints` in the above example). 

## Pretrained Weights
We release the pretrained weights of FreGrad model trained on LJSpeech for 1M steps at this [link](https://drive.google.com/drive/folders/1sOLFglnoGsUusSl5rBr_K7m82Y4RVBK9?usp=sharing). Please download and extract the file to checkpoints directory to achieve a directory as follow: 
```bash
checkpoints/
   | fregrad/
      | weights-1000000.pt
      | params_saved.py

   | stats_priorgrad/
```

`stats_priorgrad` saved at `data_root` is required to use the checkpoint for training and inference. Refer to the step 3 of the [Quick Start and Examples](#quick-start-and-examples) above.


The codebase defines `weights.pt` as a symbolic link of the latest checkpoint.
Restore the link with `ln -s weights-1000000.pt weights.pt` to continue training (`__main__.py`), or perform inference (`inference.py`) without specifying `--step`

## References
Our backbone code is based on following opensource:
- [Official implementation of code PriorGrad-vocoder](https://github.com/microsoft/NeuralSpeech/tree/master/PriorGrad-vocoder)

> We give thanks to reference open-sources for kindly publish their code for research community.

## Citations
If you find FreGrad useful to your work, please consider citing the paper as below:

      @INPROCEEDINGS{fregrad,
      author={Tan Dat Nguyen, Ji-Hoon Kim, Youngjoon Jang, Jaehun Kim, Joon Son Chung},
      booktitle={ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
      title={FreGrad: Lightweight and fast frequency-aware diffusion vocoder}, 
      year={2024},
      }

