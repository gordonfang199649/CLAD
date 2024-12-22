import numpy as np
import soundfile as sf
import os
from torch.utils import data
import pandas as pd

class RawAudio(data.Dataset):
    def __init__(self, path_to_database, meta_csv, nb_samp = 0, cut = True, return_label = True, norm_scale = True, part='train'):
        super(RawAudio, self).__init__()
        self.nb_samp = nb_samp
        self.path_to_audio = path_to_database
        self.labels = {"spoof": 1, "bonafide": 0}
        self.cut = cut
        self.return_label = return_label
        self.norm_scale = norm_scale
        self.part = part
        if self.cut and self.nb_samp == 0: raise ValueError('when adjusting utterance length, "nb_samp" should be input')
        # 讀取 meta.csv 文件
        meta_path = os.path.join(path_to_database, part, meta_csv)
        self.meta_data = pd.read_csv(meta_path)
    
    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, index):
        
        file = self.meta_data.iloc[index]['filename']
        label = self.meta_data.iloc[index]['label']
        filepath = os.path.join(self.path_to_audio, self.part, 'audio', file)
        try:
            X, _ = sf.read(filepath) 
            X = X.astype(np.float64)
        except:
            raise ValueError('%s'%filepath)

        if self.norm_scale:
            X = self._normalize_scale(X).astype(np.float32)
        X = X.reshape(1,-1) #because of LayerNorm for the input

        if self.cut:
            nb_time = X.shape[1]
            if nb_time > self.nb_samp:
                start_idx = np.random.randint(low = 0, high = nb_time - self.nb_samp)
                X = X[:, start_idx : start_idx + self.nb_samp][0]
            elif nb_time < self.nb_samp:
                nb_dup = int(self.nb_samp / nb_time) + 1
                X = np.tile(X, (1, nb_dup))[:, :self.nb_samp][0]
            else:
                X = X[0]
        if not self.return_label:
            return X
        y = self.labels[label]
        return X, y 

    def _normalize_scale(self, x):
        '''
        Normalize sample scale alike SincNet.
        '''
        if x.size == 0:
            raise ValueError("The input array 'x' is empty.")
        return x/np.max(np.abs(x))