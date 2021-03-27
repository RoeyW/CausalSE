import numpy as np
import tensorflow as tf
import datetime as dt
import csv
from tensorflow.keras import optimizers
from CausalST.CausalStation import CausalStation
from CausalST.Causal_nl import CausalNL
from CausalST.CausalStation import CausalStation
from CausalST.dataInput import load_train_Data,load_test_data
from CausalST.Metrics import real_mae
import os

from sklearn.metrics import mean_squared_error

gcn_dim,rnn_dim = 20,4
learning_rate= 1e-3
cluster_dim = 30
epoch_num =50
batch_size = 16
tr_num=49
demand_tag = "start"


current_time = dt.datetime.now().strftime("%Y%m%d-%H%M%S")

train_loss = tf.keras.metrics.Mean(name='train_loss')
test_err = tf.keras.metrics.Mean(name='test_error')

train_log_dir = 'logs\\gradient_tape\\' + current_time + '\\train'
test_log_dir = 'logs\\gradient_tape\\' + current_time + '\\test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)




def Emb_reg(new_conf_emb, id):
    # compare original embedding and missing solved embedding
    # sample: positive treatment and have his data,  remove his data
    # find simulated missing samples
    # index = tf.where(id > 0)[:, 0]
    # simulated_sample = tf.constant([], dtype=tf.float32)
    # if tf.equal(tf.size(id), 0):
    #     missing_num = 0
    # else:
    #     simulated_sample = tf.gather(new_conf_emb, index)
    #     missing_num = int(tf.size(index) / 2)
    if np.size(id)>0:
        missing_num = int(np.size(id)/2)
        simulated_sample = tf.gather(new_conf_emb, id)
        diff = simulated_sample[:missing_num] - simulated_sample[missing_num:]
        reg = tf.reduce_mean(tf.nn.l2_normalize(diff, axis=-1), -1)  # [missing_num,dim]
        return tf.reduce_mean(reg)
    else:
        return 0



tf.random.set_seed(1)

def exp(a,b,w_file):
    # data generator
    generator = load_train_Data(demand_tag, batch_size, epoch_num)
    # if act =='tanh':
    #     act_function = tf.nn.tanh
    # if act == 'sig':
    #     act_function = tf.nn.sigmoid
    MODEL = CausalNL(cluster_dim=cluster_dim, rnn_dim=rnn_dim, hid_dim=gcn_dim,act=tf.nn.sigmoid)

    # train
    for epoch in range(epoch_num):
        if epoch < 20:
            optimizer = optimizers.Adam(learning_rate)
        else:
            optimizer = optimizers.Adam(learning_rate * 0.1)
        for n in range(tr_num):
            # Input = [tr, cluster,loc,his, groundtruth]
            INPUT, sim_index = next(generator)
            date_list = INPUT[:, 0]
            INPUT_TENSOR = tf.convert_to_tensor(INPUT[:, 2:], dtype=tf.float32)

            with tf.GradientTape() as tape:
                pred, tr_pred, mse_loss, treat_loss, new_conf, effect_weight = MODEL(INPUT_TENSOR)

                # loss and gradient
                emb_reg = Emb_reg(new_conf, sim_index)
                loss_v = tf.reduce_mean(mse_loss + a * treat_loss) + b* emb_reg

                grads = tape.gradient(loss_v, MODEL.trainable_variables)
                optimizer.apply_gradients(zip(grads, MODEL.trainable_variables))
                train_loss(loss_v)
                # print('Epoch:',str(epoch + 1),'Sample:',n,loss_v)

        print('Epoch:', str(epoch + 1), 'Loss:', train_loss.result())
        # with train_summary_writer.as_default():
        #     tf.summary.scalar('loss', train_loss.result(), step=epoch)
        train_loss.reset_states()

    # MODEL.save('saveModel\\myModelinCiti_nonlinear')

    # test
    # MODEL = tf.keras.models.load_model('saveModel\\myModelinCiti_nonlinear')
    test_generator = load_test_data(demand_tag)
    mse_full = []
    for i in range(14):
        TEST_INPUT = next(test_generator)
        test_label = np.reshape(TEST_INPUT[:, -1], [-1, 1])
        test_index = np.array(TEST_INPUT[:, 1], dtype=np.int)
        test_tr = np.reshape(np.array(TEST_INPUT[:, 2], dtype=np.float), [-1, 1])

        TEST_INPUT_TENSOR = tf.convert_to_tensor(TEST_INPUT[:, 2:], dtype=tf.float32)
        test_pred, tr_pred, test_mse_loss, test_treat_loss, _, w_e = MODEL.predict(TEST_INPUT_TENSOR)

        treat_path = w_file + 'treat\\'
        if not os.path.exists(treat_path):
            os.makedirs(treat_path)
        treat_w = treat_path+ str(i) + '.csv'

        test_tr_ed = np.where(test_tr > -1, 1, 0)
        tr_pred = np.reshape(tr_pred, [-1, 1])
        treat_mae = np.abs(test_tr * test_tr_ed - tr_pred * test_tr_ed)
        np.savetxt(treat_w, treat_mae, delimiter=',')

        effect_path = w_file+'nl_w\\'
        if not os.path.exists(effect_path):
            os.makedirs(effect_path)
        test_effect_w = effect_path + str(i) + '.csv'
        np.savetxt(test_effect_w, w_e, delimiter=',')

        test_mae_w = w_file + str(i) + '.csv'
        test_mse = real_mae(test_label, test_pred, test_index, test_mae_w)
        mse_full.append(test_mse)
        print('test_err', i, ':', test_mse)

        mean_mse = np.mean(test_mse_loss)
        test_err(mean_mse)

    #     with test_summary_writer.as_default():
    #         tf.summary.scalar('test_mse', data=mean_mse, step=i)
    #
    # with test_summary_writer.as_default():
    #     tf.summary.scalar('test_mse_final', data=test_err.result(), step=0)

    mse_f = w_file + 'mse.csv'
    mse_arr = np.array(mse_full)
    np.savetxt(mse_f, mse_arr, delimiter=',')
    #
    # train_summary_writer.close()
    # test_summary_writer.close()

a_list = [0.1,0.01,1e-3,1e-4]
b_list = [0,0.1,0.01,1e-3,1e-4]
# act = ['tanh','sig']
for i in range(len(a_list)):
    for j in range(len(b_list)):
        # w_file = "D:\my\mypaper\CausalBike\EXP\\capital\\CAUSALST_nl\\insig_" +str(i)+str(j) + '\\'
        if not os.path.exists(w_file):
            os.makedirs(w_file)
        exp(a_list[i],b_list[j], w_file)
