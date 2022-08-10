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
import os
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
            if not os.path.exists('data/noise'):
                print('No existing noise data file found. Create data from command background noise')
                self._make_background_noise_list()
            self.noise_data = open('data/noise').readlines()
            self.noise_data = [i.replace('\n', '') for i in self.noise_data]
                
            print('Dataloader performs with noise augmentation using noises from "noise" file. Noise added at 50 dB')
            print('NOTE: If desired to change noise decibel, please modify _wav2fbank with desire noise level.')
            
        ## class label
        self.Label2Indx = {}
        for i, cls in enumerate(self.audio_conf.get('labels')):
            self.Label2Indx[cls] = i
           
        
    def _make_background_noise_list(self,):
        tmp = eval(self.dataset[0])['audio_filepath']
        for i in range(2):
            tmp = os.path.split(tmp)[0]

        bgn = os.listdir(os.path.join(tmp, '_background_noise_'))
        with open(os.path.join('data', 'noise'), 'w') as fout:
            for item in bgn:
                if 'wav' in item:
                    print(os.path.join(tmp, '_background_noise_', item), file=fout)
                    
    def _add_snr_noise(self, audio, noise, snr):
        noise_len = noise.size(1)
        if noise_len < audio.size(1):
            rand_start = np.random.randint(audio.size(1) - noise_len)
            noise_tmp = torch.zeros_like(audio)
            noise_tmp[:, rand_start: rand_start+noise_len] = noise

        elif noise_len > audio.size(1):
            rand_start = np.random.randint(noise_len - audio.size(1))
            noise_tmp = noise = noise[:, rand_start: rand_start + audio.size(1)]

        else:
            noise_tmp = noise

        audio_power = audio.norm(p=2)
        noise_power = noise.norm(p=2)

        snr = 10 ** (snr/20)
        scale = snr * noise_power/ audio_power
        augmented = (scale * audio + noise_tmp) / 2

        return augmented
    
    def _spec_augmentation(self, x,
                           num_time_mask=1,
                           num_freq_mask=1,
                           max_time=25,
                           max_freq=25):

        """perform spec augmentation 
        Args:
            x: input feature, T * F 2D
            num_t_mask: number of time mask to apply
            num_f_mask: number of freq mask to apply
            max_t: max width of time mask
            max_f: max width of freq mask
        Returns:
            augmented feature
        """
        max_freq_channel, max_frames = x.size()

        # time mask
        for i in range(num_time_mask):
            start = random.randint(0, max_frames - 1)
            length = random.randint(1, max_time)
            end = min(max_frames, start + length)
            x[:, start:end] = 0

        # freq mask
        for i in range(num_freq_mask):
            start = random.randint(0, max_freq_channel - 1)
            length = random.randint(1, max_freq)
            end = min(max_freq_channel, start + length)
            x[start:end, :] = 0

        return x
    
    
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
            
            # add noise here
            sample_noise = np.random.choice(self.noise_data)
            noise_wav = torchaudio.load(sample_noise)[0]
            add_with_snr = 50   ## if desire for random range of snr, consider -> np.random.choice(np.arange(low, high + 1))
            mix_waveform = self._add_snr_noise(mix_waveform, noise_wav, add_with_snr)
            
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
        if self.audio_conf.get('mode') == 'train':
            fbank = self._spec_augmentation(fbank, max_freq=25, max_time=25)

        # normalize the input
        fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        
        ## random time shift +/- 10 frame
        if self.audio_conf.get('mode') == 'train':
            fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)

        # the output fbank shape is [time_frame_num, frequency_bins], e.g., [98, 64]
        return fbank, label_indices

    def __len__(self):
        return len(self.dataset)
