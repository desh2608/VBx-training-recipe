#!/bin/bash

# Copyright      2017 Johns Hopkins University (Author: Daniel Povey)
#                2017 Johns Hopkins University (Author: Daniel Garcia-Romero)
#                2017 David Snyder
#                2022 Desh Raj
# Apache 2.0
#
# This script is similar to get_egs.sh, but it additionally does egs preparation for
# multi-speaker data, i.e., each input sample can have 1 or 2 labels.


# Begin configuration section.
cmd=run.pl
# each archive has data-chunks off length $frames_per_eg.
frames_per_chunk=200

not_used_frames_percentage=10

num_diagnostic_archives=1  # we want to test the training likelihoods
                           # on a range of utterance lengths, and this number controls
                           # how many archives we evaluate on.


compress=true   # set this to false to disable compression (e.g. if you want to see whether
                # results are affected).

num_heldout_utts=100     # number of utterances held out for training subset

num_repeats=1 # number of times each speaker repeats per archive
num_archives=-1

stage=0
nj=6         # This should be set to the maximum number of jobs you are
             # comfortable to run in parallel; you can increase it if your disk
             # speed is greater and you have more machines.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 2 ]; then
  echo "Usage: $0 [opts] <data> <egs-dir>"
  echo " e.g.: $0 data/train exp/xvector_a/egs"
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config file containing options"
  echo "  --nj <nj>                                        # The maximum number of jobs you want to run in"
  echo "                                                   # parallel (increase this only if you have good disk and"
  echo "                                                   # network speed).  default=6"
  echo "  --cmd (utils/run.pl;utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --frames-per-eg <#frames;200>                    # The number of frames per chunk that we dump"
  echo "  --not-used-frames-percentage <10>                # The percentage of frames that will ignored."
  echo "                                                   # The archives will incrementally generated and it will "
  echo "                                                   # stop when percentage of not used frames reaches to"
  echo "                                                   # this number."
  echo "  --num-archives <#num_archives;-1>                # The maximum number ark files which will be created."
  echo "                                                   # If it was set to -1 (default value) the number of ark"
  echo "                                                   # files will be estimated based on the other parameters."
  echo "  --num-repeats <#repeats;1>                       # The (approximate) number of times the training"
  echo "                                                   # data is repeated in the egs"
  echo "  --num-diagnostic-archives <#archives;3>          # Option that controls how many different versions of"
  echo "                                                   # the train and validation archives we create (e.g."
  echo "                                                   # train_subset.{1,2,3}.egs and valid.{1,2,3}.egs by default;"
  echo "                                                   # they contain different utterance lengths."
  echo "  --stage <stage|0>                                # Used to run a partially-completed training process from somewhere in"
  echo "                                                   # the middle."

  exit 1;
fi

data=$1
dir=$2

for f in ${data}/utt2num_frames ${data}/feats.scp ; do
  [ ! -f ${f} ] && echo "$0: expected file ${f}" && exit 1;
done

feat_dim=$(feat-to-dim scp:${data}/feats.scp -) || exit 1

mkdir -p ${dir}/info ${dir}/info ${dir}/temp
temp=${dir}/temp

echo ${feat_dim} > ${dir}/info/feat_dim
echo '0' > ${dir}/info/left_context
# The examples have at least frames_per_chunk right context.
echo ${frames_per_chunk} > ${dir}/info/right_context
echo '1' > ${dir}/info/frames_per_eg
cp ${data}/utt2num_frames ${dir}/temp/utt2num_frames

if [ ${stage} -le 0 ]; then
  echo "$0: Preparing train and validation lists"
  # Pick a list of heldout utterances for validation egs
  awk '{print $1}' ${data}/utt2spk | utils/shuffle_list.pl | head -${num_heldout_utts} > ${temp}/valid_uttlist || exit 1;
  # The remaining utterances are used for training egs
  utils/filter_scp.pl --exclude ${temp}/valid_uttlist ${temp}/utt2num_frames > ${temp}/utt2num_frames.train
  utils/filter_scp.pl ${temp}/valid_uttlist ${temp}/utt2num_frames > ${temp}/utt2num_frames.valid
  # Pick a subset of the training list for diagnostics
  awk '{print $1}' ${temp}/utt2num_frames.train | utils/shuffle_list.pl | head -${num_heldout_utts} > ${temp}/train_subset_uttlist || exit 1;
  utils/filter_scp.pl ${temp}/train_subset_uttlist <${temp}/utt2num_frames.train > ${temp}/utt2num_frames.train_subset

  # Create a mapping from utterance to speaker ID (an integer) or pair of ids
  python local/nnet3/xvector/create_spk_int_map.py ${data}
  mv ${data}/utt2int ${temp}/utt2int

  utils/filter_scp.pl ${temp}/utt2num_frames.train ${temp}/utt2int > ${temp}/utt2int.train
  utils/filter_scp.pl ${temp}/utt2num_frames.valid ${temp}/utt2int > ${temp}/utt2int.valid
  utils/filter_scp.pl ${temp}/utt2num_frames.train_subset ${temp}/utt2int > ${temp}/utt2int.train_subset
fi

# num_pdfs=$(awk '{print $2}' ${temp}/utt2int | sort | uniq -c | wc -l)
num_pdfs=$(wc -l < ${data}/spk2int)

# first for the training data... work out how many archives.
num_train_frames=$(awk '{n += $2} END{print n}' <${temp}/utt2num_frames.train)
num_train_subset_frames=$(awk '{n += $2} END{print n}' <${temp}/utt2num_frames.train_subset)

echo ${num_train_frames} > ${dir}/info/num_frames
echo ${num_diagnostic_archives} > ${dir}/info/num_diagnostic_archives

echo "$0: Producing archives for training. Number of archives will estimated based on the training data."

if [ ${stage} -le 2 ]; then
  echo "$0: Allocating training examples"
  ${cmd} ${dir}/log/allocate_examples_train.log \
    sid/nnet3/xvector/allocate_egs_multi.py \
      --num-repeats=${num_repeats} \
      --frames-per-chunk=${frames_per_chunk} \
      --num-pdfs=${num_pdfs} \
      --not-used-frames-percentage=${not_used_frames_percentage} \
      --num-archives=${num_archives} \
      --num-jobs=${nj} \
      --utt2len-filename=${dir}/temp/utt2num_frames.train \
      --utt2int-filename=${dir}/temp/utt2int.train --egs-dir=${dir}  || exit 1

  echo "$0: Allocating training subset examples"
  ${cmd} ${dir}/log/allocate_examples_train_subset.log \
    sid/nnet3/xvector/allocate_egs_multi.py \
      --prefix train_subset \
      --num-repeats=3 \
      --frames-per-chunk=${frames_per_chunk} \
      --num-pdfs=${num_pdfs} --num-jobs=1 \
      --num-archives=${num_diagnostic_archives} \
      --utt2len-filename=${dir}/temp/utt2num_frames.train_subset \
      --utt2int-filename=${dir}/temp/utt2int.train_subset --egs-dir=${dir}  || exit 1

  # echo "$0: Allocating validation examples"
  # ${cmd} ${dir}/log/allocate_examples_valid.log \
  #  sid/nnet3/xvector/allocate_egs_multi.py \
  #    --prefix valid \
  #    --num-repeats=3 \
  #    --frames-per-chunk=${frames_per_chunk} \
  #    --num-pdfs=${num_pdfs} --num-jobs=1 \
  #    --num-archives=${num_diagnostic_archives} \
  #    --utt2len-filename=${dir}/temp/utt2num_frames.valid \
  #    --utt2int-filename=${dir}/temp/utt2int.valid --egs-dir=${dir}  || exit 1
fi

# At this stage we'll have created the ranges files that define how many egs
# there are and where they come from.  If this is your first time running this
# script, you might decide to put an exit 1 command here, and inspect the
# contents of exp/${dir}/temp/ranges.* before proceeding to the next stage.
if [ ${stage} -le 3 ]; then
  # The script assumes you've prepared the features ahead of time.
  feats="scp,s,cs:utils/filter_scp.pl ${temp}/ranges.JOB ${data}/feats.scp |"
  train_subset_feats="scp,s,cs:utils/filter_scp.pl ${temp}/train_subset_ranges.1 ${data}/feats.scp |"
  # valid_feats="scp,s,cs:utils/filter_scp.pl ${temp}/valid_ranges.1 ${data}/feats.scp |"

  echo "$0: Generating training examples on disk"
  rm ${dir}/.error 2>/dev/null
  for g in $(seq ${nj}); do
    outputs=$(awk '{for(i=1;i<=NF;i++)printf("ark:%s ",$i);}' ${temp}/outputs.${g})
    ${cmd} ${dir}/log/train_create_examples.${g}.log \
      nnet3-xvector-get-egs-multi --compress=${compress} --num-pdfs=${num_pdfs} ${temp}/ranges.${g} \
      "`echo ${feats} | sed s/JOB/${g}/g`" ${outputs} || touch ${dir}/.error &
  done
  train_subset_outputs=$(awk '{for(i=1;i<=NF;i++)printf("ark:%s ",$i);}' ${temp}/train_subset_outputs.1)
  echo "$0: Generating training subset examples on disk"
  ${cmd} ${dir}/log/train_subset_create_examples.1.log \
    nnet3-xvector-get-egs-multi --compress=${compress} --num-pdfs=${num_pdfs} ${temp}/train_subset_ranges.1 \
    "${train_subset_feats}" ${train_subset_outputs} || touch ${dir}/.error &
  wait
  #valid_outputs=$(awk '{for(i=1;i<=NF;i++)printf("ark:%s ",$i);}' ${temp}/valid_outputs.1)
  #echo "$0: Generating validation examples on disk"
  #${cmd} ${dir}/log/valid_create_examples.1.log \
  #  nnet3-xvector-get-egs --compress=${compress} --num-pdfs=${num_pdfs} ${temp}/valid_ranges.1 \
  #  "${valid_feats}" ${valid_outputs} || touch ${dir}/.error &
  #wait
  if [ -f ${dir}/.error ]; then
    echo "$0: Problem detected while dumping examples"
    exit 1
  fi
fi

num_train_archives=$(cat ${dir}/info/num_archives)

if [ ${stage} -le 4 ]; then
  echo "$0: Shuffling order of archives on disk"
  ${cmd} --max-jobs-run ${nj} JOB=1:${num_train_archives} ${dir}/log/shuffle.JOB.log \
    nnet3-shuffle-egs --srand=JOB ark:${dir}/egs_temp.JOB.ark \
    ark,scp:${dir}/egs.JOB.ark,${dir}/egs.JOB.scp || exit 1;
  ${cmd} --max-jobs-run ${nj} JOB=1:${num_diagnostic_archives} ${dir}/log/train_subset_shuffle.JOB.log \
    nnet3-shuffle-egs --srand=JOB ark:${dir}/train_subset_egs_temp.JOB.ark \
    ark,scp:${dir}/train_diagnostic_egs.JOB.ark,${dir}/train_diagnostic_egs.JOB.scp || exit 1;
  #${cmd} --max-jobs-run ${nj} JOB=1:${num_diagnostic_archives} ${dir}/log/valid_shuffle.JOB.log \
  #  nnet3-shuffle-egs --srand=JOB ark:${dir}/valid_egs_temp.JOB.ark \
  #  ark,scp:${dir}/valid_egs.JOB.ark,${dir}/valid_egs.JOB.scp || exit 1;
fi

if [ ${stage} -le 5 ]; then
  for file in $(for x in $(seq ${num_diagnostic_archives}); do echo ${dir}/train_subset_egs_temp.${x}.ark; done) \
    $(for x in $(seq ${num_diagnostic_archives}); do echo ${dir}/valid_egs_temp.${x}.ark; done) \
    $(for x in $(seq ${num_train_archives}); do echo ${dir}/egs_temp.${x}.ark; done); do
    [ -L ${file} ] && rm $(readlink -f ${file})
    rm ${file}
  done
  rm -rf ${dir}/valid_diagnostic.scp ${dir}/train_diagnostic.scp
  for x in $(seq ${num_diagnostic_archives}); do
    cat ${dir}/train_diagnostic_egs.${x}.scp >> ${dir}/train_diagnostic.scp
    cat ${dir}/valid_egs.${x}.scp >> ${dir}/valid_diagnostic.scp
  done
  ln -sf train_diagnostic.scp ${dir}/combine.scp
fi

echo "$0: Finished preparing training examples"