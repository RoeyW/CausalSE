# CausalStation without missing fix
import tensorflow as tf
import tensorflow.keras.layers as layers
from CausalST.GCN_layer import GCN_layer

# use community emb and station's information(location,his data)
# didn't align community emb and fix missing data

class CST_WM(tf.keras.Model):

    def __init__(self,cluster_dim,hid_dim,rnn_dim):
        super(CST_WM,self).__init__()
        # cluster embedding [40]
        self.cluster_dim = cluster_dim
        gcn_dim_list = [cluster_dim, hid_dim]
        self.GCN_layers = GCN_layer(hid_dim_list=gcn_dim_list, layer_num=1, activation=tf.nn.leaky_relu)
        # station embedding
        self.his_gru = layers.GRU(rnn_dim)
        self.loc_Dense = layers.Dense(2)
        self.cl_Dense = layers.Dense(rnn_dim)

        # weights from cluster to station
        self.station_dim = tf.cast(rnn_dim + 2, tf.float32)

        conf_dim = 2 * rnn_dim + 2
        self.rnn_dim = rnn_dim
        self.mTc_Dense = layers.Dense(rnn_dim)

        # Confonduers to treatments
        self.treat_Dense = layers.Dense(1)

        # effect parameters
        self.ew_Dense = layers.Dense(1)
        self.conf_Dense = layers.Dense(1)
        self.no_treatd_Dense = layers.Dense(1)

    def embCluster(self, c_input, s2c_ids):
        c_matrix = tf.reshape(c_input, [-1, self.cluster_dim, self.cluster_dim])
        gcn_out = self.GCN_layers(c_matrix)
        cfs_emb = tf.nn.embedding_lookup(gcn_out, s2c_ids)
        return cfs_emb

    def embStation(self, s_input):
        his_data = s_input[0]
        loc_data = s_input[1]
        his_emb = self.his_gru(his_data)
        loc_emb = self.loc_Dense(loc_data)
        st_emb = tf.concat([loc_emb, his_emb], axis=-1)
        return st_emb

    def concatSaC(self, cl_emb, st_emb):
        # cfs_emb_update = tf.matmul(cfs_emb,self.cTs_weight)+self.cTs_bias
        comb_emb = tf.concat([cl_emb, st_emb], axis=-1)
        return comb_emb

    def allocateC2S(self, cl_emb, st_emb):
        # cfs_emb = tf.nn.embedding_lookup(cl_emb,s2c_ids)
        cfs_emb_hat = self.cl_Dense(cl_emb)
        # C' = matmul(C,W) dimension trans
        # cfs_emb_update = tf.matmul(cfs_emb,self.cTs_weight)+self.cTs_bias
        # C'*S/sqrt(dim)*C'
        # cfs_unit = tf.expand_dims(tf.math.reduce_sum(cfs_emb_update*st_emb,axis=-1)/tf.sqrt(self.station_dim),-1)*cfs_emb_update
        cfs_unit = tf.nn.l2_normalize(cfs_emb_hat, axis=-1)
        # st_emb_norm = tf.nn.l2_normalize(st_emb,axis=-1)
        comb_emb = tf.concat([cfs_unit, st_emb], axis=-1)
        return comb_emb

    def mapMissing(self, cl_emb, st_emb, his_data):
        # complete embedding for missing features of samples
        # find row indexes which exist missing data
        non_zero = tf.math.count_nonzero(his_data, axis=1, dtype=tf.int32)
        miss_mask = tf.where(tf.equal(non_zero, 0), 1.0, 0.0)  # [batch_size,1]
        # his_sum = tf.reduce_sum(his_data,axis=1)
        # miss_mask = tf.where(tf.equal(his_sum,0),1.0,0.0) #[batch_size,1]
        # if there are missing samples,some rows = 1; else all = 0
        #
        loc_emb = st_emb[:, 0:2]
        # cl_emb = tf.nn.l2_normalize(cl_emb,axis=-1)

        comb_emb = tf.concat([cl_emb, loc_emb], axis=-1)

        # miss_emb = comb_emb*miss_mask
        # m_hat = (m*W+b) + m= [batch, rnn]
        miss_fix = self.mTc_Dense(comb_emb) * miss_mask

        his_emb = st_emb[:, 2:]  # rnn out
        # non_miss = comb_emb[:, :-self.rnn_dim]
        miss_hat = miss_fix + his_emb

        loc_emb = tf.nn.l2_normalize(loc_emb, axis=-1)
        miss_hat = tf.nn.l2_normalize(miss_hat, axis=-1)
        new_st = tf.concat([loc_emb, miss_hat], axis=-1)
        # miss_emb_hat = tf.concat([non_miss,miss_hat],axis=-1)
        #
        # inv_mask = 1-miss_mask
        # comp_emb = miss_his*inv_mask
        # new_emb = comp_emb+miss_emb_hat
        return new_st

    def predTreat(self, conf_emb):
        Treat_pred = self.treat_Dense(conf_emb)
        return Treat_pred

    def MSE_loss(self, label, pred):
        return tf.keras.losses.mean_squared_error(label, pred)

    def treat_loss(self, treat, treat_pred):
        treated = tf.where(treat > -1, 1.0, 0.0)
        treat_ed = treat * treated
        pred_ed = treat_pred * treated
        return tf.keras.losses.mean_squared_error(treat_ed, pred_ed)

    def predict_Out(self, conf, tr):
        # treated/contrl stations should have different weights
        treated_mask = tf.zeros_like(tr)
        treated_mask = tf.where(tf.less(tr, 0), treated_mask, 1.0)
        ctrl_mask = 1.0 - treated_mask

        v = self.ew_Dense(conf) * treated_mask
        tr_emb = tr * v
        treated_out = self.conf_Dense(conf) + tr_emb
        com_emb = tf.concat([conf, tr], axis=-1)
        ctrl_out = self.no_treatd_Dense(com_emb)
        out = treated_out * treated_mask + ctrl_out * ctrl_mask

        # out = tf.matmul(S_tr_emb, self.Effect_weights) + self.Effect_bias
        return out, v

    def call(self, inputs, **kwargs):
        # Input = [ tr, s2c_id, cluster,loc,his, groundtruth]
        tr = tf.expand_dims(inputs[:, 0], -1)
        s2c_ids = tf.cast(inputs[:, 1], dtype=tf.int32)  # [batch_size]
        cluster_rel = inputs[:, 2:1602]
        loc = inputs[:, 1602:1604]
        his_data = tf.expand_dims(inputs[:, 1604:-1], -1)
        label = tf.expand_dims(inputs[:, -1], -1)

        # get cluster embedding
        Cl_emb = self.embCluster(cluster_rel, s2c_ids)

        # get station embedding
        St_emb = self.embStation((his_data, loc))

        # trans the missing his sample
        new_st = self.mapMissing(Cl_emb, St_emb, his_data)

        # fuse cluster embedding and station embedding
        new_conf = self.allocateC2S(Cl_emb, new_st)

        # predict treatment
        treat_pred = self.predTreat(new_conf)

        # predict demand
        outputs, v = self.predict_Out(new_conf, tr)
        return outputs, treat_pred, self.MSE_loss(label, outputs), self.treat_loss(tr, treat_pred), new_conf, v
