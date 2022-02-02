#!/bin/bash

if [ "$#" -ne 2 ]; then
  echo "Illegal number of parameters. Expecting egs directory and output directory."
  exit 1
fi

egs_dir=$1
out_dir=$2

# linear | arc_margin | sphere | add_margin | add_margin_multi
metric="add_margin_multi"
model="ResNet101"
resume="exp/raw_81.pth"
embed_dim=256

. ./path.sh
. parse_options.sh

utils/queue-freegpu.pl -l "hostname=c*" --gpu 1 --mem 10G $out_dir/train.log \
  python local/train_pytorch_dnn.py --model ${model} \
      --model-init ${resume} \
      --num-targets 1211 \
      --dir ${out_dir}/${model}_embed${embed_dim} \
      --metric ${metric} \
      --egs-dir ${egs_dir} \
      --minibatch-size 16 \
      --embed-dim ${embed_dim} \
      --warmup-epochs 0 \
      --initial-effective-lrate 0.0001 \
      --final-effective-lrate 0.00001 \
      --initial-margin-m 0.1 \
      --final-margin-m 0.1 \
      --optimizer SGD \
      --momentum 0.9 \
      --optimizer-weight-decay 0.0001 \
      --preserve-model-interval 20 \
      --num-epochs 2 \
      --apply-cmn no \
      --fix-margin-m 2 \
      --multi-speaker \
      --cleanup

