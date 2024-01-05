# the following command performs test set inference of PriorGrad-vocoder with default parameters defined in params.py
# inference requires the automatically generated params_saved.py during training, which is located at model_dir. 
# need to specify model_dir, data_root, and test filelist
python inference.py \
checkpoints/fregrad \
checkpoints \
filelists/test.txt \
--step 1000000 \
# --fast \
# --fast_iter 6
