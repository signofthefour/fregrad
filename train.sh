# the following command trains FreGrad-vocoder with default parameters defined in params.py
# need to specify model_dir, data_root, and training filelist
CUDA_VISIBLE_DEVICES=0 python __main__.py \
checkpoints/fregrad_test \
checkpoints/ \
filelists/train.txt
