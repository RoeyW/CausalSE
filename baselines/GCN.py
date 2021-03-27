import tensorflow as tf
import tensorflow.keras.layers as layers
from CausalST.GCN_layer import GCN_layer

# use community emb and station's information(location,his data)
# didn't align community emb and fix missing data

class GCN_GRU(tf.keras.Model):

    def __init__(self,cluster_dim,hid_dim,rnn_dim):
        super(GCN_GRU,self).__init__()
        gcn_dim_list = [cluster_dim, hid_dim]
        self.cluster_dim = cluster_dim
        self.Cluster_GCN =GCN_layer(hid_dim_list=gcn_dim_list,layer_num=1,activation=tf.nn.relu)
        self.RNN_LAYER = layers.GRU(rnn_dim,kernel_initializer=tf.keras.initializers.glorot_normal(1))
        self.loc_Dense = layers.Dense(2,kernel_initializer=tf.keras.initializers.glorot_normal(1))
        self.cl_Dense = layers.Dense(rnn_dim,kernel_initializer=tf.keras.initializers.glorot_normal(1))
        self.tr_Effect_weights = self.add_weight(shape=[2*rnn_dim + 3, 1], name='treated effect weights',
                                                 initializer=tf.keras.initializers.glorot_normal(1), trainable=True)
        self.tr_Effect_bias = self.add_weight(shape=[1], name='treated effect bias', initializer=tf.keras.initializers.glorot_normal(1),
                                              trainable=True)
        self.ctrl_Effect_weights = self.add_weight(shape=[2*rnn_dim + 3, 1], name='control effect weights',
                                                   initializer=tf.keras.initializers.glorot_normal(1), trainable=True)
        self.ctrl_Effect_bias = self.add_weight(shape=[1], name='control effect bias', initializer=tf.keras.initializers.glorot_normal(1),
                                                trainable=True)

    def predict_Out(self,S_tr_emb,tr):
        # treated/contrl stations should have different weights
        treated_mask = tf.zeros_like(tr)
        treated_mask = tf.where(tf.less(tr,0),treated_mask,1.0)
        ctrl_mask = 1.0-treated_mask
        treated_out = tf.matmul(S_tr_emb,self.tr_Effect_weights)+self.tr_Effect_bias
        ctrl_out = tf.matmul(S_tr_emb,self.ctrl_Effect_weights)+self.ctrl_Effect_bias
        out = treated_out*treated_mask + ctrl_out*ctrl_mask

        # out = tf.matmul(S_tr_emb, self.Effect_weights) + self.Effect_bias
        return out

    def call(self, inputs, **kwargs):
        tr = tf.expand_dims(inputs[:, 0], -1)
        s2c_ids = tf.cast(inputs[:, 1], dtype=tf.int32)  # [batch_size]
        cluster_rel = inputs[:, 2:1602]
        loc = inputs[:, 1602:1604]
        his_data = tf.expand_dims(inputs[:, 1604:-1], -1)

        # cluster embed
        cluster_rel = tf.reshape(cluster_rel,[-1,self.cluster_dim,self.cluster_dim])
        cl_emb = self.Cluster_GCN(cluster_rel)
        cfs_emb = tf.nn.embedding_lookup(cl_emb, s2c_ids)
        c_new_emb = self.cl_Dense(cfs_emb)

        loc_emb = tf.nn.l2_normalize(self.loc_Dense(loc),axis=-1)
        his_emb = tf.nn.l2_normalize(self.RNN_LAYER(his_data),axis=-1)

        st_emb = tf.concat([loc_emb,his_emb],axis=-1)

        c_new_emb = tf.nn.l2_normalize(c_new_emb,axis=-1)
        comb_emb = tf.concat([c_new_emb,st_emb],axis=-1)
        comb_emb = tf.concat([comb_emb,tr],axis=-1)
        output = self.predict_Out(comb_emb,tr)
        return output,self.tr_Effect_weights
