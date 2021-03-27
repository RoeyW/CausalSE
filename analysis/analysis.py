import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

new_station = pd.read_csv('D:\my\dataset\citibike\data\\new_stations.csv',header=None).values
demand_start = pd.read_csv('D:\my\dataset\citibike\data\day\demand\\demand_Spart.csv',header=0).values
demand_end = pd.read_csv('D:\my\dataset\citibike\data\day\demand\\demand_Epart.csv',header=0).values
dis_ss = pd.read_csv('D:\my\dataset\citibike\data\\distGraph_abs.csv',header=None)
addTIme = pd.read_csv('D:\\my\\dataset\\citibike\\data\\day\\demand\\addTime.csv')
Station2Cluster = pd.read_csv('D:\\my\\dataset\\citibike\\data\\station_cluster40.csv', header=None).values

# citibike: 2016/10/1 - 2017/10/31
# capitalbike: 2011/1/1 - 2012/12/31

def treatSametime():
    date = '2017-01-09'
    sametime_df = addTIme.groupby('date').count()
    oneInsametime = sametime_df[sametime_df['id']==1]
    # dates when only add one station
    one_date = oneInsametime.index
    id = addTIme[addTIme['date']==date]['id'].values[0]
    return id,date


# demand changing before and after two weeks new station added
def demandCh():
    # find index of new stations
    # ns_id = np.nonzero(new_station)[0]
    # # find stations around the new station
    # one_ns = ns_id[4]
    ns_id,tr_date = treatSametime()
    # dis_gr = dis_ss[ns_id]
    # sur_id = np.where(dis_gr<=200)[0]
    # print(sur_id)

    # find the time the new station added
    ns_d = demand_start[ns_id]
    ns_noz = np.where(ns_d>0)[0]
    add_time = ns_noz[0]

    # demand change of new stations
    # time duration [add_time-1, add_time+6]
    start_time = add_time-14
    end_time = add_time+21
    ns_mean = []
    for i in range(start_time,end_time,7):
        ns_mean.append( np.mean(ns_d[i:i+7]))
    plt.plot(ns_mean)
    plt.legend(['new'])
    plt.show()

    # existing stations around new station
    c_id = np.where(Station2Cluster[ns_id]>0)[0]
    c_s = Station2Cluster[:,c_id]
    sur_id = np.where(c_s>0)[0]
    print(c_id)

    for s_i in sur_id:
        es_mean = []
        if s_i==ns_id: continue
        sur_se = demand_start[s_i,:]
        for j in range(start_time,end_time,7):
            es_mean.append(np.mean(sur_se[j:j+7]))
        plt.plot(es_mean)
    plt.show()

def clusterDemandch(c_id,date):
    f_path = 'D:\my\dataset\citibike\data\week\\demand\\in_demand\\'+date+'.csv'
    # historical demand
    s_v = pd.read_csv(f_path,header=None).values
    s_v = s_v/7
    CS_m = np.transpose(Station2Cluster,[1,0])
    Cv_f = np.matmul(CS_m,s_v)
    C_v = Cv_f[:,-3:]
    C_max = np.expand_dims(np.max(C_v,axis=-1),axis=-1)
    C_min = np.expand_dims(np.min(C_v,axis=-1),axis=-1)
    diff = (C_max-C_min)+1
    min_m = np.ones_like(C_v)*C_min
    C_norm = (C_v-min_m)/diff
    # cluster with new station
    # plt.plot(C_norm[c_id])

    # cluster without new station
    plt.figure(figsize=(4,3))
    plt.rc('font',size=12)
    plt.plot(C_norm[1],marker='o',markersize=6,linewidth=1.5) # interact
    plt.plot(C_norm[28],marker='o',markersize=6,linewidth=1.5) # no interact
    plt.legend(['Community A','Community B'])
    plt.xticks([0,1,1.5,2],['Week 1','Week 2','Add new','Week 3'],rotation=60)
    plt.axvline(x=1.5,linestyle = '--',color='r')
    plt.ylabel('Demands')
    plt.tight_layout()
    plt.show()


def statistics():
    input_dir = 'D:\my\dataset\citibike\data\\tr_input\in\\'
    files = os.listdir(input_dir)
    es_num = 0
    for f in files:
        f_name = input_dir+f
        df = pd.read_csv(f_name)
        e_s = df[df['treat']>-1]
        es_num+=e_s.shape[0]
    print(es_num)

def treat_max(path):
    max=0
    files = os.listdir(path)
    for f in files:
        name = path+f
        v = np.genfromtxt(name,delimiter=',')
        temp = np.max(v[:,2])
        if max< temp:
            max = temp
    print(max)
# treat_max('D:\my\dataset\citibike\data\Treatment\\')
# statistics()
# demandCh()
clusterDemandch(23,'2017-01-09')
# treatSametime()