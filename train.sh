#!/bin/bash

export LANG=zh_CN.UTF-8
export LC_ALL=C.UTF-8
export LANGUAGE=zh_CN:en_US:en
work_dir=$(dirname $0)
cd $work_dir

python train_aegan.py \
--path data/ffhq \
--sample_cache data/sample_ffhq128_64.npy \
--iter 200000 \
--size 128 \
--name ffhq_aegan_wplus_joint \
--which_latent w_plus \
--lambda_rec_w 0 \
--log_every 500 \
--save_every 2000 \
--eval_every 2000 \
--dataset imagefolder \
--inception inception_ffhq128.pkl \
--n_sample_fid 10000 \
--lambda_rec_d 0.1 \
--lambda_fake_d 0.9 \
--lambda_fake_g 0.9 \
--joint \  # joint train G with D
--g_reg_every 0 \
--batch 16 \
--lr 0.0025 \
--r1 0.2048 \
--ema_kimg 5 \
--which_metric fid_sample fid_recon --use_adaptive_weight --disc_iter_start 30000