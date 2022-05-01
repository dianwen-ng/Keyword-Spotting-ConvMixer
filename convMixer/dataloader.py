#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Created on Oct 10, 2021, 17:23:06
# @modified from author: Yuan Gong
# @code reference: https://github.com/YuanGongND/ast/blob/master/src/dataloader.py
# @author: dianwen ng

import torchaudio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional
from torch.utils.data import Dataset
import random

class AudiosetDataset(Dataset):
    def __init__(self, dataset_json, audio_conf):
        """
        Dataset that manages audio recordings
        :param audio_config: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json i.e. manifest.json
        """
        self.dataset = open(dataset_json).readlines()
        self.audio_conf = audio_conf
        
        print('---------------Building {:s} dataloader---------------'.format(self.audio_conf.get('mode')))
        
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.freqm = self.audio_conf.get('freq_masks')
        self.timem = self.audio_conf.get('time_masks')
        print('Using following mask: {:d} freq, {:d} time'.format(self.audio_conf.get('freq_masks'), self.audio_conf.get('time_masks')))
        
        self.mixup = self.audio_conf.get('mixup')
        print('Using mix-up with rate {:f}'.format(self.mixup))
        
        # dataset spectrogram mean and std, used to normalize the input
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')
        print('Use dataset mean {:.3f} and std {:.3f} for normalization'.format(self.norm_mean, self.norm_std))
        
        # if add noise for data augmentation
        self.noise = self.audio_conf.get('noise')
        if self.noise == True:
            print('Dataloader performs with noise augmentation')
            
        ## class label
        self.Label2Indx = {
            'unknown': 0,
            'silence': 1,
            'yes':     2,
            'no':      3,
            'up':      4,
            'down':    5,
            'left':    6,
            'right':   7,
            'on':      8,
            'off':     9,
            'stop':    10,
            'go':      11}
        
    def _wav2fbank(self, filename1, filename2=None):
        # no mixup
        if filename2 == None:
            waveform, sr = torchaudio.load(filename1)
            waveform = waveform - waveform.mean()
                
        # with mixup
        else:
            waveform1, sr = torchaudio.load(filename1)
            waveform2, _ = torchaudio.load(filename2)

            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # do padding to equal length
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0, 0:waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # trim longer audio to equal length
                    waveform2 = waveform2[0, 0:waveform1.shape[1]]
                    
            # sample lambda from beta distribtion
            mix_lambda = np.random.beta(10, 10)

            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()

        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, 
                                                  sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=self.melbins, 
                                                  dither=0.0, frame_shift=10) 

        # pad and trunction (fbank)
        max_length = self.audio_conf.get('target_length')
        n_frames = fbank.shape[0]
        p = max_length - n_frames
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:max_length, :]

        if filename2 == None:
            return fbank, 0
        else:
            return fbank, mix_lambda

    
    def __getitem__(self, index):
        """
        return: audio, label
        audio is a FloatTensor of size (BS, N_frames, N_freq) for spectrogram
        label is an integer
        """
        audio_meta = eval(self.dataset[index])
        
        ## do mix-up for this sample (controlled by the given mixup rate)
        if random.random() < self.mixup:
            # find another sample to mix, also do balance sampling
            mix_sample_idx = random.randint(0, len(self.dataset)-1)
            mix_audio_meta = eval(self.dataset[mix_sample_idx])
            
            # get the mixed fbank
            fbank, mix_lambda = self._wav2fbank(audio_meta['audio_filepath'],
                                                mix_audio_meta['audio_filepath'])
            
            # initialize the label
            label_indices = np.zeros(len(self.Label2Indx))
            label_indices[self.Label2Indx[audio_meta['command']]] += mix_lambda
            label_indices[self.Label2Indx[mix_audio_meta['command']]] += 1.0-mix_lambda
            label_indices = torch.FloatTensor(label_indices)
        
        # skip mixup
        else:
            label_indices = np.zeros(len(self.Label2Indx))
            fbank, mix_lambda = self._wav2fbank(audio_meta['audio_filepath'])
            label_indices[self.Label2Indx[audio_meta['command']]] = 1.0
            label_indices = torch.FloatTensor(label_indices)

        ## perform SpecAug, not do for eval set
        freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        timem = torchaudio.transforms.TimeMasking(self.timem)
        fbank = torch.transpose(fbank, 0, 1)
        
        if self.freqm != 0:
            fbank = freqm(fbank)
        
        if self.timem != 0:  
            fbank = timem(fbank)
        
        fbank = torch.transpose(fbank, 0, 1)

        # normalize the input
        fbank = (fbank - self.norm_mean) / (self.norm_std * 2)

        # perform noise perturbation with random white noise
        if self.noise == True:
            fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            
            ## random time shift +/- 10 frame
            fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)

        # the output fbank shape is [time_frame_num, frequency_bins], e.g., [98, 64]
        return fbank, label_indices

    def __len__(self):
        return len(self.dataset)