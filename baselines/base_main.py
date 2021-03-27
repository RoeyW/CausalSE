import tensorflow as tf
import numpy as np
import os
from CausalST.baselines.GRU import GRU
from CausalST.baselines.GCN import GCN_GRU
from CausalST.dataInput import load_train_Data,load_test_data
from CausalST.Metrics import real_mae
from CausalST.baselines.SCEG import SCEG
from CausalST.baselines.Missing import CST_WM



gcn_dim,rnn_dim = 20,4
learning_rate= 1e-3
cluster_dim = 30
epoch_num =50
batch_size = 16
tr_num=49
demand_tag = "end"
w_file = 'D:\my\mypaper\CausalBike\EXP\capital\\GRU\\out\\'
if not os.path.exists(w_file): os.makedirs(w_file)

tf.random.set_seed(-1)
# MODEL = SCEG(cluster_dim,gcn_dim,rnn_dim)
MODEL = GRU(rnn_dim)


generator = load_train_Data(demand_tag,batch_size,epoch_num)

def loss_function(label, pred):
    return tf.reduce_mean(tf.keras.losses.mean_squared_error(label, pred))


final_effect_weight=0.0

for epoch in range(epoch_num):
    if epoch<25:
        optimizer = tf.keras.optimizers.Adam(learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate*0.1)
    for n in range(tr_num):
        # Input = [tr, cluster,loc,his, groundtruth]
        INPUT,sim_index = next(generator)
        date_list = INPUT[:, 0]
        INPUT_TENSOR = tf.convert_to_tensor(INPUT[:, 2:], dtype=tf.float32)
        label = tf.reshape(INPUT_TENSOR[:,-1],[-1,1])

        with tf.GradientTape() as tape:
            pred, effect_weight = MODEL(INPUT_TENSOR)
            final_effect_weight = effect_weight[-1].numpy()

            # loss and gradient
            loss_v = loss_function(label,pred)

            grads = tape.gradient(loss_v, MODEL.trainable_variables)
            optimizer.apply_gradients(zip(grads, MODEL.trainable_variables))

effect_f = w_file+'effect.csv'
np.savetxt(effect_f,final_effect_weight,delimiter=',')


#test
test_generator = load_test_data(demand_tag)
mse_full=[]

tr_all=[]
pred_all=[]
prev_all = []
for i in range(14):
    TEST_INPUT = next(test_generator)
    test_label = np.reshape(TEST_INPUT[:,-1],[-1,1])
    test_index = np.array(TEST_INPUT[:,1],dtype=np.int)
    test_tr = np.array(TEST_INPUT[:,2],dtype=np.float)

    TEST_INPUT_TENSOR = tf.convert_to_tensor(TEST_INPUT[:, 2:], dtype=tf.float32)
    test_pred,_= MODEL.predict(TEST_INPUT_TENSOR)

    pred_all.extend(test_pred)
    tr_all.extend(test_tr)
    prev_all.extend(TEST_INPUT[:,-2])



    test_mae_w = w_file+str(i)+'.csv'
    test_mse =real_mae(test_label,test_pred,test_index,test_mae_w)
    print('test ',i,':',test_mse)
    mse_full.append(test_mse)

pred_all_arr =np.reshape( np.array(pred_all),[-1,1])
tr_all_arr = np.reshape(np.array(tr_all),[-1,1])
prev_all_arr = np.reshape(np.array(prev_all),[-1,1])
diff_arr = pred_all_arr-prev_all_arr
all_res = np.concatenate([tr_all_arr,diff_arr],axis=-1)
all_f = w_file+'IDeffect.csv'
np.savetxt(all_f,all_res,delimiter=',')

mse_f = w_file+'mse.csv'
mse_arr = np.array(mse_full)
np.savetxt(mse_f,mse_arr,delimiter=',')