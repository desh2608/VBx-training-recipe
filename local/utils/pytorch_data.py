import os
import queue
import struct
import subprocess
from threading import Thread

import numpy as np
import torch

from torch.utils.data import Dataset



class KaldiArkLoader(object):

    def __init__(self, ark_file, apply_cmn, queue_size=1280, multi=False):
        self._ark_file = ark_file
        self._queue = queue.Queue(queue_size)
        self._reading_finished = False
        self._killed = False
        self._apply_cmn = apply_cmn
        self._thread = Thread(target=self._load_data)
        self._thread.daemon = True
        self._thread.start()
        self.multi = multi

    def _load_data(self):
        def read_token(_fid):
            token = ''
            while True:
                char = _fid.read(1).decode('utf-8')
                if char == '' or char == ' ':
                    break
                token += char
            return token

        def cleanup(_process, _cmd):
            ret = _process.wait()
            if ret > 0:
                raise Exception('cmd %s returned %d !' % (_cmd, ret))
            return

        cmn_command = ""
        if self._apply_cmn:
            cmn_command = " | apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 ark:- ark:-"
        cmd = "nnet3-copy-egs-to-feats --print-args=false ark:{} ark:-{}".format(self._ark_file, cmn_command)
        fh = open("/dev/null", "w")
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=fh)
        cleanup_thread = Thread(target=cleanup, args=(process, cmd))
        cleanup_thread.daemon = True
        cleanup_thread.start()
        fid = process.stdout

        key = read_token(fid)
        cnt = 0
        # note: here for speed we assume that the size if 4 bytes, so we do not check others
        while key:
            # print(key)
            _, _, _, rows, _, cols = struct.unpack('=bibibi', fid.read(15))
            buf = fid.read(rows * cols * 4)
            vec = np.frombuffer(buf, dtype=np.float32)
            mat = np.reshape(vec, (rows, cols))
            # extract speaker label(s) from key
            if not self.multi or key.startswith("single"):
                label = int(key.split("-")[-1])
            else:
                label1, label2 = int(key.split("-")[-2]), int(key.split("-")[-1])
                label = (label1, label2)
            self._queue.put((mat, label, key))
            cnt += 1
            # self._queue.append((mat, label, key))
            if self._killed:
                # stop reading from this file
                break
            key = read_token(fid)
        fid.close()
        fh.close()
        self._reading_finished = True
        # print('Reading finished. Count: %d' % cnt)

    def next(self, timeout=900):
        if self._reading_finished and self._queue.empty():
            return None, None, None
        next_element = self._queue.get(block=True, timeout=timeout)
        # print('Queue size: %d' % self._queue.qsize())
        return next_element

    def stop_reading(self):
        # print('stop_reading was called')
        self._killed = True
        # here we just removing unprocessed items from the queue
        # to make garbage collector happy.
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Empty:
                break


class KaldiArkDataset(Dataset):
    def __init__(self, egs_dir, num_archives, num_workers, rank, num_examples_in_each_ark, apply_cmn,
                 finished_iterations=0, processed_archives=0, prefix=''):
        super(KaldiArkDataset, self).__init__()
        self.data_loader = None
        self.egs_dir = egs_dir
        self.num_archives = num_archives
        self.num_workers = num_workers
        self.rank = rank
        self.apply_cmn = apply_cmn
        self.finished_iterations = finished_iterations
        self.processed_archives = processed_archives
        self.num_examples_in_each_ark = num_examples_in_each_ark
        self.length = num_workers * num_examples_in_each_ark
        self.prefix = prefix

    def __getitem__(self, idx):
        # read the next example from the corresponding loader
        mat, spk_id, name = self.data_loader.next()
        return mat, spk_id

    def __len__(self):
        return self.length

    def set_iteration(self, iteration):
        assert iteration > self.finished_iterations
        iteration -= self.finished_iterations
        ark_idx = (self.processed_archives + (iteration - 1) *
                   self.num_workers + self.rank) % self.num_archives + 1
        ark_file = os.path.join(self.egs_dir, f'{self.prefix}egs.{ark_idx}.ark')
        assert os.path.isfile(ark_file), f'Path to ark with egs `{ark_file}` not found.'
        if self.data_loader is not None:
            # first stop the running thread
            self.data_loader.stop_reading()
        self.data_loader = KaldiArkLoader(ark_file, self.apply_cmn)

class KaldiArkDatasetMultiSpeaker(Dataset):
    def __init__(self, egs_dir, num_targets, num_archives, num_workers, rank, num_examples_in_each_ark, apply_cmn,
                 finished_iterations=0, processed_archives=0, prefix=''):
        super(KaldiArkDatasetMultiSpeaker, self).__init__()
        self.data_loader = None
        self.egs_dir = egs_dir
        self.num_targets = num_targets
        self.num_archives = num_archives
        self.num_workers = num_workers
        self.rank = rank
        self.apply_cmn = apply_cmn
        self.finished_iterations = finished_iterations
        self.processed_archives = processed_archives
        self.num_examples_in_each_ark = num_examples_in_each_ark
        self.length = num_workers * num_examples_in_each_ark
        self.prefix = prefix

    def __getitem__(self, idx):
        # read the next example from the corresponding loader
        mat, spk_id, name = self.data_loader.next()
        # convert spk_id to one-hot vector
        spk_id_1hot = torch.zeros(self.num_targets)
        if isinstance(spk_id, int):
            spk_id_1hot[spk_id] = 1
        else:
            spk_id_1hot[spk_id[0]] = 1
            spk_id_1hot[spk_id[1]] = 1
        return mat, spk_id_1hot

    def __len__(self):
        return self.length

    def set_iteration(self, iteration):
        assert iteration > self.finished_iterations
        iteration -= self.finished_iterations
        ark_idx = (self.processed_archives + (iteration - 1) *
                   self.num_workers + self.rank) % self.num_archives + 1
        ark_file = os.path.join(self.egs_dir, f'{self.prefix}egs.{ark_idx}.ark')
        assert os.path.isfile(ark_file), f'Path to ark with egs `{ark_file}` not found.'
        if self.data_loader is not None:
            # first stop the running thread
            self.data_loader.stop_reading()
        self.data_loader = KaldiArkLoader(ark_file, self.apply_cmn, multi=True)

