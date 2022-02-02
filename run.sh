#!/bin/bash

#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2020 Phonexia
# Author: Jan Profant <jan.profant@phonexia.com>
# All Rights Reserved

nnet_dir=exp/xvector_nnet
stage=0
train_stage=-1

. ./cmd.sh || exit 1
. ./path.sh || exit 1
set -e
. ./utils/parse_options.sh

vaddir=mfcc
mfccdir=mfcc
fbankdir=fbank
plda_train_dir=data/plda_train
min_len=400
rate=16k
all_data_dir=voxceleb1_comb

# set directory for corresponding datasets
voxceleb1_path=/export/corpora5/VoxCeleb1_v2/wav
# voxceleb2_dev_path=
# voxceleb_cn_path=


if [ ${stage} -le 1 ]; then
  # prepare voxceleb1, voxceleb2 dev data and voxceleb-cn
  # parameter --remove-speakers removes test speakers from voxceleb1
  # please see script utils/make_data_dir_from_voxceleb.py to adapt it to your needs
  python utils/make_data_dir_from_voxceleb.py --out-data-dir data/voxceleb1 \
    --dataset-name voxceleb1 --remove-speakers local/voxceleb1-test_speakers.txt \
    --dataset-path ${voxceleb1_path} --rate ${rate}
  utils/fix_data_dir.sh data/voxceleb1
  utils/data/get_utt2dur.sh --cmd "$train_cmd" --nj 50 data/voxceleb1
fi

if [ ${stage} -le 2 ]; then
  # in this stage, we create multi-speaker mixed utterances from the voxceleb1 dataset.
  # For training, we will combine the single and multi-speaker utterances.
  python local/generate_multi_speaker_data.py --in-data-dir data/voxceleb1 \
    --out-data-dir data/voxceleb1_comb --single-speaker-utts 70000
  utils/fix_data_dir.sh data/voxceleb1_comb
fi

if [ ${stage} -le 3 ]; then
  # in this stage, we compute VAD and prepare features

  # make mfccs from clean audios (will be only used to compute vad afterwards)
  steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc_${rate}.conf --nj 100 --cmd \
    "${feats_cmd}" data/${all_data_dir} exp/make_mfcc ${mfccdir}
  utils/fix_data_dir.sh data/${all_data_dir}

  # compute VAD for clean audio
  local/compute_vad_decision.sh --nj 100 --cmd \
    "queue.pl --mem 2G -l hostname=!b02*" data/${all_data_dir} exp/make_vad ${vaddir}
  utils/fix_data_dir.sh data/${all_data_dir}

  # make fbanks from clean audios
  steps/make_fbank.sh --write-utt2num-frames true --fbank-config conf/fbank_${rate}.conf --nj 100 --cmd \
    "${feats_cmd}" data/${all_data_dir} exp/make_fbank ${fbankdir}
  utils/fix_data_dir.sh data/${all_data_dir}
fi

name=${all_data_dir}
if [ ${stage} -le 4 ]; then
  # Now we prepare the features to generate examples for xvector training.
  # This script applies CMVN and removes nonspeech frames.  Note that this is somewhat
  # wasteful, as it roughly doubles the amount of training data on disk.  After
  # creating training examples, this can be removed.
  local/nnet3/xvector/prepare_feats_for_egs.sh --nj 100 --cmd "${train_cmd}" \
    data/${name} data/${name}_no_sil exp/${name}_no_sil
  utils/fix_data_dir.sh data/${name}_no_sil

  # Now, we need to remove features that are too short after removing silence
  # frames.  We want at least 4s (400 frames) per utterance.
  mv data/${name}_no_sil/utt2num_frames data/${name}_no_sil/utt2num_frames.bak
  awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' data/${name}_no_sil/utt2num_frames.bak > data/${name}_no_sil/utt2num_frames
  utils/filter_scp.pl data/${name}_no_sil/utt2num_frames data/${name}_no_sil/utt2spk > data/${name}_no_sil/utt2spk.new
  mv data/${name}_no_sil/utt2spk.new data/${name}_no_sil/utt2spk
  utils/fix_data_dir.sh data/${name}_no_sil

  # We also want several utterances per speaker. Now we'll throw out speakers
  # with fewer than 8 utterances.
  # min_num_utts=8
  # awk '{print $1, NF-1}' data/${name}_with_aug_no_sil/spk2utt > data/${name}_with_aug_no_sil/spk2num
  # awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' data/${name}_with_aug_no_sil/spk2num | utils/filter_scp.pl - data/${name}_with_aug_no_sil/spk2utt > data/${name}_with_aug_no_sil/spk2utt.new
  # mv data/${name}_with_aug_no_sil/spk2utt.new data/${name}_with_aug_no_sil/spk2utt
  # utils/spk2utt_to_utt2spk.pl data/${name}_with_aug_no_sil/spk2utt > data/${name}_with_aug_no_sil/utt2spk

  utils/filter_scp.pl data/${name}_no_sil/utt2spk data/${name}_no_sil/utt2num_frames > data/${name}_no_sil/utt2num_frames.new
  mv data/${name}_no_sil/utt2num_frames.new data/${name}_no_sil/utt2num_frames

  # Now we're ready to create training examples.
  utils/fix_data_dir.sh data/${name}_no_sil
fi

if [ ${stage} -le 5 ]; then
  echo "$0: Getting neural network training egs";
  local/nnet3/xvector/get_egs_multi.sh --cmd "$train_cmd" \
    --nj 50 \
    --stage 3 \
    --frames-per-chunk 400 \
    --not-used-frames-percentage 10 \
    --num-archives 500 \
    --num-diagnostic-archives 1 \
    --num-repeats 2 \
    data/${name}_no_sil exp/egs
fi

exit 0
