import numpy as np
import os
import pandas as pd

def load_train_Data(demand_tag,batch_size,epoch_num):
    input_dir=""
    if demand_tag=='start':
        # input = [tr(1),cluster(1600),station_info,demand(n,n+1)]
        input_dir ="D:\my\dataset\citibike\data\\tr_input\\in\\"
    if demand_tag == 'end':
        input_dir = "D:\my\dataset\citibike\data\\tr_input\\out\\"

    file_list = os.listdir(input_dir)
    # shuffle the treatment
    train_set = file_list[:-14]*epoch_num
    pos_sample_num = int(batch_size / 2)

    np.random.shuffle(train_set)
    for f in train_set:
        # pick same number of positive and negative samples
        f_name = input_dir + f
        df = pd.read_csv(f_name)
        pos_all = df[df['treat'] > -1].values
        neg_all = df[df['treat'] == -1].values
        # shuffle pos samples and neg samples
        # pos_sample: have treatment [0,1]
        # neg_sample: no treatment [-1]
        np.random.shuffle(pos_all)
        np.random.shuffle(neg_all)
        if pos_all.shape[0] < pos_sample_num:
            pos_spl = pos_all
            neg_num = batch_size - pos_all.shape[0]
            neg_spl = neg_all[:neg_num]
        else:
            pos_spl = pos_all[:pos_sample_num]
            neg_spl = neg_all[:pos_sample_num]

        # simulate some missing data from existing stations
        train_samples = np.concatenate([pos_spl, neg_spl], axis=0)

        # pos_spl = pos_spl[np.argsort(pos_spl[:, 2])]
        # copy samples, which have no zeros
        non_zero_ct = np.count_nonzero(pos_spl[:,1606:-1],axis=1)
        non_zero_index = np.where(non_zero_ct== np.shape(pos_spl[:,1606:-1])[1])[0]
        mask_spl = pos_spl[non_zero_index]
        mask_spl[:, 1606:-1] = 0.0
        simulated_index = np.arange(np.size(non_zero_index))+batch_size
        simulated_index = np.concatenate([non_zero_index,simulated_index],axis=0)
        train_samples = np.concatenate([train_samples, mask_spl], axis=0)
        yield train_samples,simulated_index



def load_test_data(demand_tag):
    input_dir = ""
    if demand_tag == 'start':
        # input = [tr(1),cluster(1600),station_info,demand(n,n+1)]
        input_dir = "D:\my\dataset\\citibike\data\\tr_input\\in\\"
    if demand_tag == 'end':
        input_dir = "D:\my\dataset\\citibike\data\\tr_input\\out\\"

    file_list = os.listdir(input_dir)
    # shuffle the treatment
    test_set = file_list[-14:]
    for f in test_set:
        # pick same number of positive and negative samples
        f_name = input_dir + f
        df = pd.read_csv(f_name)
        test_sample = df.values
        yield test_sample


