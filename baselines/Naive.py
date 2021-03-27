from CausalST.dataInput import load_test_data
import numpy as np
from CausalST.Metrics import real_mae
import os

# predict_old_station = mean (his data)
# predict_new_station = mean(around)

generator = load_test_data("end")
base_path = 'D:\my\mypaper\CausalBike\EXP\capital\\NAIVE\\out\\'
if not os.path.exists(base_path): os.makedirs(base_path)
final_mse = []
final_eff=[]
final_tr = []
for i in range(14):
    # Input = [date,s_id, tr, s2c_id, cluster,loc,his, groundtruth]
    test_data= next(generator)
    s_index = np.array(test_data[:,1],dtype=np.int)
    tr = test_data[:,2]
    his = test_data[:,-906:-1]
    prev = his[:,-1]
    label = test_data[:,-1]
    # all stations' pred = mean(his)
    pred = np.mean(his,axis=-1)
    new_id = np.where(tr==0)[0]
    for n in new_id:
        new_c = test_data[n,3]
        same_c = np.where(test_data[:,3]==new_c)[0]
        SinC = np.mean(pred[same_c])
        pred[n] = SinC
    w_f = base_path+str(i)+'.csv'
    mse = real_mae(label,pred,s_index,w_f)
    final_mse.append(mse)
    eff = pred-prev
    final_eff.extend(eff)
    final_tr.extend(tr)

mse_f = base_path+'mse.csv'
mse_arr = np.array(final_mse)
np.savetxt(mse_f,mse_arr,delimiter=',')

eff_f = base_path+'IDeffect.csv'
eff_arr = np.reshape(np.array(final_eff),[-1,1])
tr_arr = np.reshape(np.array(final_tr),[-1,1])
out = np.concatenate([tr_arr,eff_arr],axis=-1)
np.savetxt(eff_f,out,delimiter=',')

