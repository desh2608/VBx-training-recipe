#!/bin/local/env python
# -*- coding: utf-8 -*-
# Copyright 2022  Johns Hopkins University (Author: Desh Raj)
# Apache 2.0
#
# This script generates the data directory for multi-speaker data.
#
# Usage:
#  $ python local/generate_multi_speaker_data.py \
#      --in-data-dir data/voxceleb1 \
#      --out-data-dir data/voxceleb1_multi

import argparse
import random
import logging
import sys
from pathlib import Path
from collections import namedtuple
from itertools import groupby

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

Utterance = namedtuple("Utterance", ["id", "speaker", "duration", "path", "type"])


def read_args():
    parser = argparse.ArgumentParser(
        description="Generate multi-speaker data directory"
    )
    parser.add_argument(
        "--in-data-dir", type=str, required=True, help="path to input data directory"
    )
    parser.add_argument(
        "--out-data-dir", type=str, required=True, help="path to output data directory"
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--single-speaker-utts",
        type=int,
        default=0,
        help="number of single speaker utterances to include in final output dir",
    )
    args = parser.parse_args()
    return args


def generate_multi_speaker_data(in_dir, out_dir):
    # First we read all utterances
    utterances = []
    logging.info("Reading utterances from %s", in_dir)
    with open(in_dir / "utt2spk", "r") as f_utt2spk, open(
        in_dir / "utt2dur", "r"
    ) as f_utt2dur, open(in_dir / "wav.scp", "r") as f_wav:
        for line1, line2, line3 in zip(f_utt2spk, f_utt2dur, f_wav):
            utt1, speaker = line1.rstrip().split(" ")
            utt2, duration = line2.rstrip().split(" ")
            assert utt1 == utt2
            utt3, path = line3.rstrip().split(" ")
            assert utt1 == utt3
            utterances.append(Utterance(utt1, speaker, float(duration), path, "single"))

    # Group utterances by speaker
    logging.info("Grouping utterances by speaker")
    utterances.sort(key=lambda x: x.speaker)
    spk_to_utts = {
        speaker: list(group)
        for speaker, group in groupby(utterances, key=lambda x: x.speaker)
    }

    mixed_utts = []
    logging.info("Generating mixed utterances")
    while len(spk_to_utts) > 1:
        # Pick 2 random speakers
        spk1, spk2 = random.sample(list(spk_to_utts.keys()), k=2)
        # print(spk1, spk2)
        # Pick a random utterance for each speaker
        utt1 = random.choice(spk_to_utts[spk1])
        utt2 = random.choice(spk_to_utts[spk2])
        # Remove the utterance from the speaker's group
        spk_to_utts[spk1].remove(utt1)
        spk_to_utts[spk2].remove(utt2)
        # Keep shorter utterance first
        if utt1.duration > utt2.duration:
            utt1, utt2 = utt2, utt1

        utt1_id = "_".join(utt1.id.split("_")[1:])
        utt2_id = "_".join(utt2.id.split("_")[1:])
        utt_id = f"{utt1.speaker}_{utt2.speaker}-{utt1_id}_{utt2_id}"
        spk_id = f"{utt1.speaker}_{utt2.speaker}"
        # We trim the mixed waveform to the shorter one
        path = f"sox -m {utt1.path} {utt2.path} -t wav -b 16 -e signed -c 1 - | sox - -t wav -b 16 -e signed -c 1 - trim 0 {utt1.duration} |"

        mixed_utts.append(Utterance(utt_id, spk_id, utt1.duration, path, "multi"))
        # If the speaker has no more utterances, remove the speaker
        if len(spk_to_utts[spk1]) == 0:
            del spk_to_utts[spk1]
        if len(spk_to_utts[spk2]) == 0:
            del spk_to_utts[spk2]

    single_utts = random.sample(utterances, k=args.single_speaker_utts)

    # Combine single and mixed utterances and sort by utterance id
    logging.info("Combining single and mixed utterances")
    utts = sorted(single_utts + mixed_utts, key=lambda x: f"{x.type}-{x.id}")

    logging.info(f"Writing {len(utts)} utterances to {out_dir}")
    with open(out_dir / "utt2spk", "w") as f_utt2spk, open(
        out_dir / "wav.scp", "w"
    ) as f_wav, open(out_dir / "utt2dur", "w") as f_utt2dur:
        for utt in utts:
            f_utt2spk.write(f"{utt.type}-{utt.id} {utt.type}-{utt.speaker}\n")
            f_wav.write(f"{utt.type}-{utt.id} {utt.path}\n")
            f_utt2dur.write(f"{utt.type}-{utt.id} {utt.duration}\n")


if __name__ == "__main__":
    args = read_args()

    # set random seed
    random.seed(args.seed)

    in_dir = Path(args.in_data_dir)
    out_dir = Path(args.out_data_dir)
    out_dir.mkdir(exist_ok=True)
    generate_multi_speaker_data(in_dir, out_dir)
