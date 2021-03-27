import pandas as pd
import numpy as np
import csv
import datetime
import os
import scipy.sparse as sp
import math
from geopy import distance






# find the time of adding stations
# station id: timestamp
def addTimestamp(new_stations,demand,date,base_path):
    # station demands from the first timestamp
    cols = ['id','date']
    w_path = base_path+"day\\demand\\addTIme.csv"
    w = csv.writer(open(w_path,'w',newline=''))
    w.writerow(cols)
    new_id = new_stations['id'].values
    for id in new_id:
        # print(id)
        station_v = demand[demand['id'] == id].values[0]
        time_i = np.nonzero(station_v)
        # print(time_i)
        s_date = date[time_i[0][0]]
        if s_date!='id':
            cols = [id,s_date]
            w.writerow(cols)


def extrHisdata(addTime,demand,demand_path):
    # At each time of adding, calculate weekly demand for each station
    # daily demand from D:\\my\\dataset\\citibike\\data\\day\\demand\\demand_Spart.csv(demand_Epart.csv)
    # write files to D:\my\dataset\citibike\data\\week\\demand
    Adddate = addTime['date'].values
    Adddate_uq = np.unique(Adddate)
    d_values =demand.values
    d_cols = demand.columns.values

    for d in Adddate_uq:
        # find new stations at one time
        add_index = np.where(d_cols == d)[0][0]
        # his_start =  (add_d-org_d)%7+org_d

        if add_index<21: continue
        his_start = add_index%7
        his_end = his_start+7
        sample_data=[]

        # write data for each treatment to each file.
        file_path = demand_path+d+'.csv'
        writer = csv.writer(open(file_path, 'w', newline=''))

        while his_end<=add_index+7:
            his_data = d_values[:,his_start:his_end]
            weekly_d = np.mean(his_data,axis=1)
            # append hisdata and target data
            sample_data.append(weekly_d)
            his_start=his_end
            his_end+=7
        sample_array =np.transpose( np.array(sample_data),[1,0])
        writer.writerows(sample_array)



def deleteSeveral(addTime,Station2Cluster,f_base_path,station_num):
    # if there are more than one new stations in one cluster at one time
    # mask the cluster, we only consider the effect when adding one new station
    # so we will not estimate the stations in the cluster, which exists more than one new stations.
    Adddate = addTime['date'].values
    Adddate_uq = np.unique(Adddate)

    for d in Adddate_uq:
        array = np.zeros(shape=(1, station_num))
        f_path = f_base_path+ d + '.csv'
        w_f = csv.writer(open(f_path, 'w', newline=''))

        series = addTime[addTime['date']==d]
        ids = series['id'].values
        for i in ids:
            array[0][i] = 1
        #     new station id * [S in C]
        s2c = np.matmul(array, Station2Cluster)
        mask = np.where(s2c>1,0,1)
        mask = np.transpose(mask,[1,0])
        s_mask = np.matmul(Station2Cluster,mask)
        w_f.writerows(s_mask)
    # print(sum_d,sum_s)


def extrTreat(addTime,demand,Station2Cluster,station_info,mask_path,tr_w_path,station_num):
    # Find the treatment when adding station
    # Treatment=[location, dis(new,old)]
    # old = find in cluster(new)
    Adddate = addTime['date'].values
    Adddate_uq = np.unique(Adddate)
    d_cols = demand.columns.values
    # find the station relationship, which stations are in the same cluster
    s2sRel = np.matmul(Station2Cluster,np.transpose(Station2Cluster,[1,0]))
    stations_location = station_info[['lat','lon']].values

    for d in Adddate_uq:
        # his_start =  (add_d-org_d)%7+org_d
        add_index = np.where(d_cols == d)[0][0]
        if add_index < 21: continue
        file_path = tr_w_path + d + '.csv'
        tr_writer = csv.writer(open(file_path, 'w', newline=''))
        series = addTime[addTime['date']==d]
        # find new station's id
        add_ids = series['id'].values

        s_msk_f = mask_path+d+'.csv'
        # mask the treatments which represent more than two stations
        s_msk = pd.read_csv(s_msk_f,header=None).values
        s2sRel = s2sRel*s_msk

        # index of new stations at one time
        new_stations = np.zeros(shape=(station_num,1))

        # add_array(station_num,2): locations of new stations
        add_array = np.zeros(shape=(station_num,2))

        # location(new) and distance(new,old)
        for id in add_ids:
            new_stations[id] = 1
            s = station_info[station_info['index_num']==id][['lat','lon']].values[0]
            add_array[id,0] = float(s[0])
            add_array[id,1] = float(s[1])
        # 0-1 new station index
        new_stations = s_msk*new_stations
        # treatment(1): new station location
        # tr(1) will be concated with stations, which are in the same cluster as new stations, including the new station
        newRelstation_location = np.matmul(s2sRel,add_array)

        # tr(2): distance(new, old)
        old_station_id = np.expand_dims(np.sum(new_stations*s2sRel,axis=0),-1)
        old_location = old_station_id*stations_location
        # dis = np.reshape(np.sum((old_location-newRelstation_location)**2,axis=-1),(-1,1))
        # geopy.distance
        dis = np.zeros(shape=(station_num,1))
        for i in range(station_num):
            if newRelstation_location[i,0] !=0:
                ns_l = newRelstation_location[i]
                os_l = old_location[i]
                dis[i] = distance.geodesic(ns_l,os_l).meters

        # treatment
        tr = np.concatenate([newRelstation_location,dis],axis=-1)
        tr_sum = np.sum(tr,axis=-1)
        # replace the no-treatment to -1
        for i in range(station_num):
            if tr_sum[i]==0:
                tr[i] = tr[i]-1
        tr_writer.writerows(tr)

def sumWeeklyStation(date, delta_days,s_path,station_num):
    file_list = os.listdir(s_path)
    end_index = file_list.index(date)
    start_index = end_index-delta_days
    week_num = delta_days/7
    # Delta_W = datetime.timedelta(days=delta_days)
    # start_date = date-Delta_W
    # Delta_D = datetime.timedelta(days=1)
    s_array = np.zeros(shape=(station_num,station_num))
    for i in range(start_index,end_index):
        f_name = s_path+ file_list[i]
        # sparse graph
        S2S_df = pd.read_csv(f_name,header=0)
        S2S_spv = S2S_df[(S2S_df['source']<station_num)&(S2S_df['target']<station_num)].values
        S2S_v = sp.coo_matrix((S2S_spv[:,3],(S2S_spv[:,1],S2S_spv[:,2])),shape=[station_num,station_num]).toarray()
        s_array+=S2S_v
    s_array = s_array/week_num
    return s_array


def clusterRelation(tr_path,s_graph_path,Station2Cluster,cw_path,station_num):
    # average weekly demands of clusters avg[t-3,t]
    # demands between cluster, check-in/out demands [C*C]
    files = os.listdir(tr_path)
    for f in files:
        c_files =cw_path+f
        writer = csv.writer(open(c_files,'w',newline=''))
        f_name = s_graph_path+f
        station_values = pd.read_csv(f_name,header=None).values
        r,c = station_values.shape

        if c <=4:
            delta_days = 7*(c-1)
        else:
            delta_days = 7*4

        s_mean = sumWeeklyStation(f,delta_days,s_graph_path,station_num)
        C2S = np.transpose(Station2Cluster,[1,0])
        c_v = np.matmul(C2S,s_mean)
        c2c = np.matmul(c_v,Station2Cluster)
        writer.writerows(c2c)

def orgInput(tr_dir,demand_dir,cluster_dir,label_mask_dir,S2C_dict,station_info,w_path,cluster_num,station_num):
    tr_fs = os.listdir(tr_dir)
    stations_raw_location = station_info[['lat', 'lon']].values
    index_num = np.reshape(station_info['index_num'].values,[-1,1])
    stations_location = normlocation(stations_raw_location)
    s2c_id = np.transpose(pd.read_csv(S2C_dict,header=None).values,[1,0])
    # some stations cannot be predicted

    for f in tr_fs:
        tr_path = tr_dir + f
        cluster_path = cluster_dir + f
        demand_path = demand_dir + f
        label_mask_path = label_mask_dir+f
        cluster_v = pd.read_csv(cluster_path, header=None).values
        demand_v = pd.read_csv(demand_path, header=None).values
        tr_v = np.expand_dims(pd.read_csv(tr_path, header=None).values[:, -1], axis=-1)
        label_mask_cr = pd.read_csv(label_mask_path,header=None).values

        max_length = demand_v.shape[-1]-1
        w_f_name = w_path+f
        w_f = open(w_f_name, 'w', newline='')
        csv_writer = csv.writer(w_f)
        # [date,treat,cluster_rel*1600,st_lat,st_lon,his_length,demand*max_length,groundtruth ]
        cols = ['date', 'index_num','treat','s2c_id']
        cl_col = ['cluster_rel' for i in range(cluster_num**2)]
        cols.extend(cl_col)
        cols.extend(['st_lat', 'st_lon'])
        # max_length of his * 'demand'
        dem_label = ['demand' for i in range(max_length)]
        cols.extend(dem_label)
        cols.append('ground_truth')
        csv_writer.writerow(cols)



        # cluster_v = np.reshape(cluster_v,[1,-1])
        cluster_postive = np.where(cluster_v>0,1,0)
        # norm cluster_adj
        norm_cluster = normClusteradj(cluster_postive)

        ones = np.ones(shape=(station_num,1))
        norm_cluster = np.reshape(norm_cluster,[1,-1])
        cluster_flat = np.matmul(ones,norm_cluster)
        # input = [date,tr(1),cluster(1600),station_info,demand(n,n+1)]
        d = f.split('.')[0]
        d_tr = [[d for i in range(1)] for j in range(station_num)]
        # [date, treat,index_num]
        input = np.concatenate([d_tr, index_num], axis=-1)
        input = np.concatenate([input,tr_v],axis=-1)
        # [date, treat,cluster]
        input = np.concatenate([input,s2c_id],axis=-1)
        input = np.concatenate([input,cluster_flat],axis=-1)
        # add st_lat, st_lon
        input = np.concatenate([input,stations_location],axis=-1)
        # add demand [maxlength: 0,0,...0,demand1,demand2]
        temp_num = max_length-(demand_v.shape[-1]-1)
        # temp = np.zeros(shape=(846,temp_num))
        # input = np.concatenate([input,temp],axis=-1)
        input = np.concatenate([input,demand_v],axis=-1)

        # remove some stations cannot be predicted
        remove_index = np.where(label_mask_cr==0)[0]
        cr_input = np.delete(input,remove_index,axis=0)
        last_input = cr_input[:,-1].astype(np.float)
        remove_index2 = np.where(last_input==0)[0]
        cr_input = np.delete(cr_input,remove_index2,axis=0)
        if cr_input.shape[0]<station_num:
            print(cr_input.shape)
        csv_writer.writerows(cr_input)

def normlocation(locs):
    # input = [lat,lon]
    max_latlon = np.max(locs,axis=0)
    min_latlon = np.min(locs,axis=0)
    diff = max_latlon-min_latlon
    new_loc = (locs-min_latlon)/diff
    new_loc = new_loc*2-1

    return new_loc

def normClusteradj(cluster_m):
    # self loop
    I = np.eye(cluster_m.shape[0])

    A_hat = I+ cluster_m
    A_hat = np.where(A_hat>1,1,A_hat)
    D = np.sum(A_hat, axis=1)
    D_inv_flat = 1 / np.sqrt(D)
    D_inv = np.diag(D_inv_flat)

    norm_A = np.matmul(D_inv, A_hat)
    norm_A = np.matmul(norm_A, D_inv)
    return norm_A

def normTreat(w_path,r_path):

    f_list = os.listdir(r_path)
    max = 0
    for f in f_list:
        f_name = r_path + f
        v = pd.read_csv(f_name).values[:, -1]
        if max < np.max(v): max = np.max(v)
    print('treatment max:', max)
    for f in f_list:
        f_name = r_path + f
        v = pd.read_csv(f_name,header=None).values[:, -1]
        v = v/max
        v = np.reshape(np.where(v<0,-1,v),[-1,1])
        w_f_name = w_path+f
        w_f = open(w_f_name,'w',newline="")
        writer = csv.writer(w_f)
        writer.writerows(v)


def findDemandNormalizer(out_demand_path,in_demand_path,norm_fname,station_num):
    w_f = open(norm_fname, 'w', newline="")
    writer = csv.writer(w_f)


    file_list = os.listdir(out_demand_path)
    min = np.array([100000] * station_num)
    max = np.array([0] * station_num)
    for f in file_list:
        f_name = out_demand_path + f
        v = pd.read_csv(f_name, header=None).values
        v_min = np.min(v, axis=-1)
        v_max = np.max(v, axis=-1)
        min = np.where(v_min < min, v_min, min)
        max = np.where(v_max > max, v_max, max)

    file_list = os.listdir(in_demand_path)
    for f in file_list:
        f_name = in_demand_path + f
        v = pd.read_csv(f_name, header=None).values
        v_min = np.min(v, axis=-1)
        v_max = np.max(v, axis=-1)
        min = np.where(v_min < min, v_min, min)
        max = np.where(v_max > max, v_max, max)
    min = np.reshape(min, [-1, 1])
    max = np.reshape(max, [-1, 1])
    min_max = np.concatenate([min, max], axis=-1)
    writer.writerows(min_max)


def normDemand(w_path,r_path,normlizer_path):
    min_max = pd.read_csv(normlizer_path,header=None).values
    min = np.reshape(min_max[:,0],[-1,1])
    max = np.reshape(min_max[:,1],[-1,1])

    file_list = os.listdir(r_path)
    for f in file_list:
        f_name = r_path + f
        v = pd.read_csv(f_name, header=None).values
        norm_v= (v-min)/(max-min)
        norm_v[np.isnan(norm_v)]=0
        norm_v[np.isinf(norm_v)]=0
        w_fname = w_path+f
        d_wf = open(w_fname,'w',newline="")
        writer = csv.writer(d_wf)
        writer.writerows(norm_v)


def main(city):
    cluster_num=0
    base_path = ""
    cluster_path=""
    if city=='nyc':
        base_path = "D:\\my\\dataset\\citibike\\data\\"
        cluster_path = base_path + "station_cluster40.csv"
        cluster_num = 40
    if city=='DC':
        base_path = "D:\\my\\dataset\\capitalbike\\data\\"
        cluster_path = base_path + "station_cluster30.csv"
        cluster_num = 30
    station_path = base_path+"station_info.csv"
    new_s_path = base_path+"new_stations.csv"
    in_demand_path = base_path+"day\\demand\\demand_Spart.csv"
    out_demand_path = base_path+"day\\demand\\demand_Epart.csv"


    station_info = pd.read_csv(station_path)
    new_stations = pd.read_csv(new_s_path, header=None)
    in_demand = pd.read_csv(in_demand_path)
    out_demand = pd.read_csv(out_demand_path)
    Station2Cluster = pd.read_csv(cluster_path, header=None).values

    station_num = station_info.values.shape[0]
    new_stations['id'] = np.arange(station_num)
    in_demand['id'] = np.arange(station_num)
    # id of new station
    new_stations = new_stations[new_stations[0] > 0]
    date = in_demand.columns.values


    # STEP 0: find addTime
    addTimestamp(new_stations,in_demand,date,base_path)
    add_tim_path = base_path+"day\\demand\\addTIme.csv"
    addTime = pd.read_csv(add_tim_path)

    # STEP 1: extract weekly data before and after treatment
    out_demand_path = base_path+'week\\demand\\out_demand\\'
    extrHisdata(addTime,out_demand,out_demand_path)
    in_demand_path = base_path+'week\\demand\\in_demand\\'
    extrHisdata(addTime,in_demand,in_demand_path)

    # STEP 2: calculate weekly data of cluster
    # D:\my\dataset\citibike\data\week\in_demand
    record_graph_path = base_path+"day\\record graph\\"
    out_crel_path = base_path +"week\\cluster_graph\\out_cluster_graph\\"
    in_crel_path = base_path + "week\\cluster_graph\\in_cluster_graph\\"
    clusterRelation(out_demand_path,record_graph_path,Station2Cluster,out_crel_path,station_num)

    clusterRelation(in_demand_path,record_graph_path,Station2Cluster,in_crel_path,station_num)

    # STEP 3: mask stations, which are in the cluster with several treatments
    mask_path = base_path + "week\\label_mask\\"
    deleteSeveral(addTime, Station2Cluster, mask_path,station_num)

    # STEP 4: extract treatment
    tr_path = base_path+"Treatment\\"
    extrTreat(addTime,in_demand,Station2Cluster,station_info,mask_path,tr_path,station_num)

    # STEP 5: normal treat and demand
    norm_tr_path = base_path+"week\\norm_treat\\"
    normTreat(norm_tr_path,tr_path)
    norm_name = base_path+"week\\demand_normalizer.csv"
    findDemandNormalizer(out_demand_path,in_demand_path,norm_name,station_num)

    in_norm_path = base_path+'week\\norm_demand\\in\\'
    out_norm_path=base_path+'week\\norm_demand\\out\\'
    normDemand(in_norm_path,in_demand_path,norm_name)
    normDemand(out_norm_path,out_demand_path,norm_name)

    # STEP 6: organize the input into one file
    # Input=[tr,cluster,his,groundtruth]
    # if no normalize, change treat, demand directory
    s2c_dict = base_path + "S2C_dict.csv"
    org_in_w_path = base_path+"\\tr_input\in\\"
    orgInput(norm_tr_path,in_norm_path,in_crel_path,mask_path,s2c_dict,station_info,org_in_w_path,cluster_num,station_num)
    org_out_w_path = base_path + "\\tr_input\out\\"
    orgInput(norm_tr_path,out_norm_path,out_crel_path,mask_path,s2c_dict,station_info,org_out_w_path,cluster_num,station_num)


# main("DC")

# base_path = "D:\\my\\dataset\\citibike\\data\\"
# station_path = base_path+"station_info.csv"
# new_s_path = base_path+"new_stations.csv"
# s2c_dict = base_path + "S2C_dict.csv"
# org_in_w_path = base_path+"\\tr_input\in\\"
# in_norm_path = base_path + 'week\\norm_demand\\in\\'
# out_norm_path = base_path + 'week\\norm_demand\\out\\'
# mask_path = base_path + "week\\label_mask\\"
# out_crel_path = base_path + "week\\cluster_graph\\out_cluster_graph\\"
# in_crel_path = base_path + "week\\cluster_graph\\in_cluster_graph\\"
# norm_tr_path = base_path+"week\\norm_treat\\"
# station_info = pd.read_csv(station_path)
# cluster_num = 40
# station_num = station_info.values.shape[0]
# orgInput(norm_tr_path,in_norm_path,in_crel_path,mask_path,s2c_dict,station_info,org_in_w_path,cluster_num,station_num)
# org_out_w_path = base_path + "\\tr_input\out\\"
# orgInput(norm_tr_path,out_norm_path,out_crel_path,mask_path,s2c_dict,station_info,org_out_w_path,cluster_num,station_num)