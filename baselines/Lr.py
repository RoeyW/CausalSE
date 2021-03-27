# Linear regression using all features

from CausalST.Metrics import real_mae
import numpy as np
from sklearn.linear_model import LinearRegression
import os
import pandas as pd

demand_tag = "start"
batch_size = 49

base_path = "D:\my\mypaper\CausalBike\EXP\\capital\\LR_all\\in\\"
if not os.path.exists(base_path): os.makedirs(base_path)

def load_train(demand_tag):
    input_dir=""
    if demand_tag == 'start':
        # input = [tr(1),cluster(1600),station_info,demand(n,n+1)]
        input_dir = "D:\my\dataset\\capitalbike\\data\\tr_input\\in\\"
    if demand_tag == 'end':
        input_dir = "D:\my\dataset\\capitalbike\\data\\tr_input\\out\\"
    file_list = os.listdir(input_dir)
    # shuffle the treatment
    train_set = file_list[:-14]
    train_x=[]
    train_y = []

    for f in train_set:
        f_name = input_dir + f
        df = pd.read_csv(f_name)
        df =df[df['treat']>-1]
        y = df['ground_truth'].values
        tr = np.reshape(df['treat'].values,[-1,1])
        loc = df[['st_lat','st_lon']].values
        demand = df.values[:, -7:-1]
        x = np.concatenate([tr,loc],axis=-1)
        x = np.concatenate([x,demand],axis=-1)

        train_x_arr = np.array(x,dtype=np.float)
        train_y_arr = np.array(y, dtype=np.float)
        train_y_arr = np.reshape(train_y_arr, [-1, 1])
        yield train_x_arr, train_y_arr


generator = load_train(demand_tag)
lr = LinearRegression()
for i in range(batch_size):
    train_x, train_y = next(generator)
    if np.size(train_y)==0: continue
    lr.fit(train_x, train_y)




# --------save effect-------------
effect_file = base_path+'effect.csv'
np.savetxt(effect_file,lr.coef_,delimiter=',')



def load_test(demand_tag):
    input_dir = ""
    if demand_tag == 'start':
        # input = [tr(1),cluster(1600),station_info,demand(n,n+1)]
        input_dir = "D:\my\dataset\capitalbike\data\\tr_input\\in\\"
    if demand_tag == 'end':
        input_dir = "D:\my\dataset\capitalbike\data\\tr_input\\out\\"
    file_list = os.listdir(input_dir)
    test_set = file_list[-14:]
    for f in test_set:
        f_name = input_dir + f
        df = pd.read_csv(f_name)
        df = df[df['treat'] > -1]
        tr = np.reshape(df['treat'].values, [-1, 1])
        loc = df[['st_lat', 'st_lon']].values
        demand = df.values[:,-7:-1]
        test_x = np.concatenate([tr, loc], axis=-1)
        test_x = np.concatenate([test_x,demand],axis=-1)
        test_y = np.expand_dims(df['ground_truth'].values,-1)
        st_index = df['index_num'].values
        prev = demand[:,-1]
        yield test_x,test_y,prev,st_index

# --------test-----------------
generator= load_test(demand_tag)
mse_full=[]
diff_final =[]
tr_final =[]
for i in range(14):
    test_x,test_y,prev,st_index =next(generator)
    if np.size(test_y)>0:
        pred = lr.predict(test_x)
        # write samples' mae results in each file
        mae_f = base_path + str(i) + '.csv'
        test_mse = real_mae(test_y, pred, st_index, mae_f)
        mse_full.append(test_mse)
        tr_final.extend(test_x[:,0])
        diff = pred-prev
        diff_final.extend(diff)


# save mse for each treatment date
mse_f = base_path+"mse.csv"
mse_arr = np.array(mse_full)
np.savetxt(mse_f,mse_arr,delimiter=",")


