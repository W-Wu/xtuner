# Copyright (c) OpenMMLab. All rights reserved.
import io
import  os

import torch

def build_ts_transform(do_normalize=True, do_truncate=True, max_len=240000):
    def transform(ts_path, sr: str):
        assert len(ts_path)==1, "Currently only one ts signal is supported."
        ts_path=ts_path[0]
        ext = os.path.splitext(ts_path)[-1].lower()
        try:
            if ext in [".wav",'.mp3','.flac']:
                try:
                    import librosa
                except ImportError:
                    raise ImportError("Please install librosa to process audio files.")
                ts_input, sr = librosa.load(ts_path, sr=None)
                ts_input = ts_input[:, None]  # [T, 1]
            elif ext == ".csv":
                try:
                    import pandas as pd
                except ImportError:
                    raise ImportError("Please install pandas to process CSV files.")
                df = pd.read_csv(ts_path, header=None)
                ts_input = df.values  # [T, C]
            elif ext == ".npy":
                try:
                    import numpy as np
                except ImportError:
                    raise ImportError("Please install numpy to process NPY files.")
                ts_input = np.load(ts_path) # [T, C]
            else:
                raise ValueError(f"Unsupported file format: {ext}")

            
            ts_tensor = torch.from_numpy(ts_input)

            if do_normalize:
                mean = ts_tensor.mean(dim=0, keepdim=True)   
                std = ts_tensor.std(dim=0, keepdim=True)  
                ts_tensor = (ts_tensor - mean) / (std + 1e-8)

            ts_tensor=ts_tensor.to(torch.bfloat16)

            if do_truncate:
                if len(ts_tensor)>240000:   # truncate to 240k to avoid oom
                    ts_tensor=ts_tensor[:240000,:]

            if len(ts_tensor.size())==1:
                ts_tensor=ts_tensor.unsqueeze(-1)
            
            ts_len=ts_tensor.size(0)
            if sr is None or sr == 0:   # if no sr provided
                sr = ts_len/4
            else:
                sr = sr[0]  # remove list

            return ts_tensor, torch.tensor(ts_len), torch.tensor(sr)
        
        except Exception as e:
            print(f"Error processing time series file {ts_path}: {e}")
            return None

    return transform