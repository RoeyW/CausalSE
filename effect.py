from sklearn.linear_model import LinearRegression
import os
import numpy as np
import pandas as pd
from CausalST.Metrics import transform
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns

path = "D:\my\dataset\citibike\data\week\\demand_normalizer.csv"
normalizer_values=pd.read_csv(path,header=None).values #[min,max]
# measure REAL Effect

def func(x,w):
    gx = 1.0 / (1+np.exp(-1.0*w * x))
    return w*gx*(1-gx)

def Effect_metric(input_dir):
    file_list = os.listdir(input_dir)
    train_set = file_list[-14:]
    x = []
    y = []
    mean_x =[]
    mean_y = []
    for f in train_set:
        f_name = input_dir + f
        df = pd.read_csv(f_name)
        st_index = df['index_num'].values
        cur_normal = normalizer_values[st_index]
        n_df = df[df['treat'] >= 0]
        # y_0 = transform(df.values[:, -2],cur_normal)
        # y_1 = transform(df['ground_truth'].values,cur_normal)

        # df['ground_truth'] = y_1
        # df['last_demand'] = y_0

        # real_label = n_df['ground_truth'].values
        # last_demand = n_df['last_demand'].values
        y_0 = n_df.values[:, -2]
        y_1 = n_df['ground_truth'].values
        tr = n_df['treat'].values
        delta_y = y_1-y_0
        x.extend(tr)
        y.extend(delta_y)

        # step = 0.01
        # for i in np.arange(0,1,step):
        #     t_df = n_df[(n_df['treat']>i)&(n_df['treat']<(i+step))]
        #     if t_df.size==0:continue
        #     mean_tr = np.mean(t_df['treat'].values)
        #
        #     yt_0 = t_df.values[:, -2]
        #     yt_1 = t_df['ground_truth'].values
        #     mean_delta_y = np.mean(np.abs(yt_1 - yt_0))
        #     mean_x.append(mean_tr)
        #     mean_y.append(mean_delta_y)
        # meanx_arr = np.array(mean_x)
        # meany_arr = np.array(mean_y)




        # real_tr = tr*max_tr

    X = np.array(x)
    # X = np.reshape(X,[-1,1])
    Y = np.array(y)
    # Y = np.reshape(Y,[-1,1])
    # nz_index = np.where(Y<0.3)
    # X = X[nz_index]
    # Y = Y[nz_index]

    # popt,popv = curve_fit(func,X,Y)
    # Y2 = [func(i,popt[0]) for i in X]
    # print(popt[0])
    # plt.plot(X, Y2, 'r')



    # XY = np.concatenate([X,Y],axis=-1)
    plt.scatter(X,Y,marker='x')

    # plt.scatter(meanx_arr,meany_arr)
    # plt.show()

    # linear regression to calculate rate for (tr, delta(y))
    # lr_rate = LinearRegression()
    # lr_rate.fit(X,Y)
    # rate = lr_rate.coef_
    # print(rate)
    return X,Y



def drawtestRate(tr_path,effect_path,pred_path):
    tr_arr = np.genfromtxt(tr_path)
    effect_arr= np.genfromtxt(effect_path)
    pred_arr = np.genfromtxt(pred_path)[:,1]


    # effect_arr = np.array(effect_set)
    id = np.where(tr_arr>0)
    tr_set = tr_arr[id]
    effect_set = effect_arr[id]
    pred_set = pred_arr[id]
    effect_set = np.where(effect_set>1,1,effect_set)
    effect_set = np.where(effect_set<-1,-0.5,effect_set)
    plt.figure(figsize=(5,4))
    plt.rc('font', size=17)
    plt.scatter(tr_set,effect_set,c='#90B44B',s=20)
    # plt.scatter(tr_set,pred_set,c='#F05E1C',s=20,marker='x')
    plt.ylim((-1,1.2))
    plt.xlim((-0.1,1))
    plt.axvline(0.4,ls='--',c='#0B1013')
    plt.xticks([0,0.4,1])
    plt.xlabel('Treatment')
    plt.ylabel('Effect')
    plt.tight_layout()
    plt.show()





def treatedMAE(tr_path,path):
    # concat all pred_mae, delete no treat out
    tr_f = os.listdir(tr_path)
    test_f = tr_f[-14:]
    tr_f = []
    treated_mae=[]
    for i in range(14):

        tr_name = tr_path + test_f[i]
        if os.path.exists(tr_name):
            df = pd.read_csv(tr_name, header=0)
            tr = df['treat'].values
            treated_mask = np.where(tr > -1, 1.0, 0.0)
            if np.sum(treated_mask) == 0: continue

            tr_f.extend(tr)
            f_name = path + str(i) + '.csv'
            mae = pd.read_csv(f_name, header=None).values[:, -1]
            treated_mae.extend(mae * treated_mask)


    out_w = path+'results.csv'
    tr_f = np.reshape(np.array(tr_f),[-1,1])
    treated_mae = np.reshape(np.array(treated_mae),[-1,1])
    out = np.concatenate([tr_f,treated_mae],axis=-1)

    np.savetxt(out_w,out,delimiter=',')


def Tr_mae(path):
    mae =pd.read_csv(path,header=None).values[:,-1]
    id = np.where(mae!=0)
    mse = np.mean(mae[id])
    print(mse)


def y_mae(path):
    arr = np.genfromtxt(path,delimiter=',')
    tr = arr[:,0]
    y_ = arr[:,1]
    id = np.where(tr>-1)[0]
    y = y_[id]
    print(np.mean(y))


def y_mae_LR(path):
    files = [0,1,2,3,5,6,8,12,13]
    s=[]
    for f in files:
        name = path+str(f)+'.csv'
        v = np.genfromtxt(name,delimiter=',')
        s.extend(v[:,1])
    s_arr = np.array(s)
    mae = np.mean(s_arr)
    print('outcome mae',mae)




def Tr_maeforNL(true_path,pred_path):
    true = np.genfromtxt(true_path,delimiter=',')
    id = np.where(true!=0)


    pred = []
    for i in range(14):
        f_name = pred_path+str(i)+'.csv'
        pred_y = np.genfromtxt(f_name,delimiter=',')
        pred.extend(pred_y)
    pred_arr = np.array(pred)
    true_c = np.reshape(true,[-1,1])
    pred_arr_c = np.reshape(pred_arr,[-1,1])
    ite_mae = np.concatenate([true_c,pred_arr_c],axis=-1)
    w_f = pred_path+'ite_mae.csv'
    np.savetxt(w_f,ite_mae)
    mean_true = np.mean(true[id])
    mean_pred = np.mean(pred_arr[id])
    eff_mae = np.abs(mean_true-mean_pred)
    print('effect mae:',eff_mae)


def Tr_effectforl(true_path,pred_path):
    true = np.genfromtxt(true_path,delimiter=',')
    id = np.where(true!=0)

    pred = np.genfromtxt(pred_path,delimiter=',')

    mean_ture = np.mean(true[id])
    eff = pred[:,1]
    pred_t = eff[id]
    mean_pred = np.mean(pred_t)

    mae_eff = np.abs(mean_ture-mean_pred)

    print('effect mae:',mae_eff)


def effect_true(tr_path,w_path):
    file_list = os.listdir(tr_path)
    train_set = file_list[-14:]
    x = []
    y = []
    for f in train_set:
        f_name = tr_path + f
        df = pd.read_csv(f_name)
        y_0 = df.values[:, -2]
        y_1 = df['ground_truth'].values
        tr = df['treat'].values
        delta_y = y_1 - y_0
        x.extend(tr)
        y.extend(delta_y)

    X = np.reshape(np.array(x),[-1,1])
    Y = np.reshape(np.array(y),[-1,1])
    # new = np.where(X == 0,1.0,0.0)
    mask= np.where(X>-1,1.0,0.0)
    new = np.where(X==0,1.0,0.0)
    new = np.reshape(new,[-1,1])

    mask = np.reshape(mask,[-1,1])
    y_new = Y*new
    y_h = Y*mask
    x_h = X*mask
    y_true = y_h/x_h
    y_true[np.isnan(y_true)]=0.0
    y_true[np.isinf(y_true)]=0.0
    y_true = y_true+y_new
    np.savetxt(w_path,y_true)

def Testtreat(tr_path):
    tr_f = os.listdir(tr_path)
    test_f = tr_f[-14:]

    tr_set = []
    for i in range(14):
        tr_name = tr_path + test_f[i]
        df = pd.read_csv(tr_name, header=0)
        tr = df['treat'].values
        tr_set.extend(tr)
    tr_arr = np.array(tr_set)
    np.savetxt('D:\my\mypaper\CausalBike\EXP\\citi\\TESTtreat.csv',tr_arr,delimiter=',')


def Typestation(tr_path,ite_path):
    # new stations: 0
    # existing stations: >0
    tr_v = np.genfromtxt(tr_path)
    ns_id = np.where(tr_v==0)
    es_id = np.where(tr_v>0)
    all_id = np.where(tr_v>-1)
    ite = np.genfromtxt(ite_path)
    ns_mae = np.mean(ite[ns_id],axis=0)
    es_mae = np.mean(ite[es_id],axis=0)
    all_mae = np.mean(ite[all_id],axis=0)
    ns_err = np.abs(ns_mae[0]-ns_mae[1])
    es_err = np.abs(es_mae[0]-es_mae[1])
    all_err = np.abs(all_mae[0]-all_mae[1])

    print('ns mae',ns_err)
    print('es mae', es_err)
    print('all mae', all_err)

def TypestationforBase(true_path,pred_path,):
    v = np.genfromtxt(pred_path,delimiter=',')
    tr_v = v[:,0]
    ns_id = np.where(tr_v == 0)
    pred_ite = v[:,1]
    true_ite = np.genfromtxt(true_path,delimiter=',')
    true_mae = np.mean(true_ite[ns_id])
    pred_mae = np.mean(pred_ite[ns_id])
    err = np.abs(true_mae-pred_mae)
    print('ns err',err)





def RangeTreat(tr_path,ite_path):
    # nyc: 0-0.3, 0.3
    # wa: 0-0.4 >0.4
    tr_v = np.genfromtxt(tr_path)
    ite = np.genfromtxt(ite_path)

    t2 = np.where((tr_v>0)&(tr_v<0.4))
    t4 = np.where(tr_v>=0.4)

    ite_2 = np.mean(ite[t2],axis=0)
    ite_4 = np.mean(ite[t4],axis=0)


    err_2 = np.abs(ite_2[0]-ite_2[1])
    err_4 = np.abs(ite_4[0]-ite_4[1])
    print(err_2,err_4)



tr_path = "D:\my\mypaper\CausalBike\EXP\\capital\\TESTtreat.csv"
w_path = 'D:\my\mypaper\CausalBike\EXP\\capital\\in_true.csv'
# drawtestRate(tr_path,w_path,'D:\my\mypaper\CausalBike\EXP\\capital\\CAUSALST_nl\\insig_34\\nl_w\\ite_mae.csv')
# Typestation(tr_path,'D:\my\mypaper\CausalBike\EXP\\capital\\CAUSALST_nl\\insig_34\\nl_w\\ite_mae.csv')
RangeTreat(w_path,'D:\my\mypaper\CausalBike\EXP\\capital\\CAUSALST_nl\\insig_34\\nl_w\\ite_mae.csv')
#
# Testtreat('D:\my\dataset\\citibike\\data\\tr_input\\in\\')


# Tr_mae(w_path)

# calculate true effect
# effect_true('D:\my\dataset\\citibike\\data\\tr_input\\in\\',w_path)

# calculate mean(individual effect)

# base_path = 'D:\my\mypaper\CausalBike\\EXP\\capital\\CAUSALST_nl\\intanh_34\\'
# tr_mae_path = base_path+'nl_w\\'
# Tr_maeforNL(w_path,tr_mae_path)
# Tr_effectforl(w_path,'D:\my\mypaper\CausalBike\EXP\\citi\\NAIVE\\in\\IDeffect.csv')
# #
# #
# tr_path = 'D:\my\dataset\\citibike\data\\tr_input\\in\\'
# treatedMAE(tr_path,base_path)
# y_mae_path = base_path+'\\results.csv'
# y_mae(y_mae_path)
# # y_mae_LR('D:\my\mypaper\CausalBike\EXP\capital\LR_all\\in\\')
