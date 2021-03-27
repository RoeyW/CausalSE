import numpy as np
import pandas as pd



path = "D:\my\dataset\citibike\data\week\\demand_normalizer.csv"
normalizer_values=pd.read_csv(path,header=None).values #[min,max]

def real_mae(label,out,st_index,w_f):
    # transform the normalized data to real value
    cur_noramlizer = normalizer_values[st_index]

    st_index = np.reshape(st_index,[-1,1])
    real_label = transform(label,cur_noramlizer)
    real_pred = transform(out,cur_noramlizer)
    mae = np.abs(real_label-real_pred)

    save_m = np.concatenate([st_index,mae],-1)
    np.savetxt(w_f,save_m,delimiter=',')

    mse = np.sqrt(np.mean(mae**2))
    return mse

def transform(norm_x,normalizer):
    norm_x = np.reshape(norm_x,[-1,1])
    diff = normalizer[:, 1] - normalizer[:, 0]
    diff = np.reshape(diff,[-1,1])
    min = np.reshape(normalizer[:,0],[-1,1])
    x = norm_x*diff+min
    return x

