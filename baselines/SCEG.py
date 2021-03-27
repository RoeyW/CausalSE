import tensorflow as tf
from BikeS.vaeTL import vaeTL
from CausalST.GCN_layer import GCN_layer


class SCEG(tf.keras.Model):
    def __init__(self,cluster_dim,rnn_dim,hid_dim):
        super(SCEG, self).__init__()
        c_hid_layer_list =[cluster_dim,hid_dim]
        self.cluster_dim = cluster_dim
        self.c_gcn_layer = GCN_layer(c_hid_layer_list, featureless=True, layer_num=1, activation=tf.nn.relu)
        self.loc_Dense = tf.keras.layers.Dense(2)
        self.RNN_layer = tf.keras.layers.GRU(rnn_dim)
        self.C_fc = tf.keras.layers.Dense(rnn_dim)
        self.VAE_layer = vaeTL(latent_dim=rnn_dim,out_dim=1)
        self.ctrl_Dense = tf.keras.layers.Dense(1)
        self.tr_Effect_weights = self.add_weight(shape=[2 * rnn_dim + 1, 1], name='treated effect weights',
                                                 initializer=tf.keras.initializers.glorot_normal(), trainable=True)
        self.tr_Effect_bias = self.add_weight(shape=[1], name='treated effect bias', initializer=tf.keras.initializers.glorot_normal(),
                                              trainable=True)

    def predict_Out(self,S_tr_emb,tr):
        # treated/contrl stations should have different weights
        treated_mask = tf.zeros_like(tr)
        treated_mask = tf.where(tf.less(tr,0),treated_mask,1.0)
        ctrl_mask = 1.0-treated_mask
        treated_out = tf.matmul(S_tr_emb,self.tr_Effect_weights)+self.tr_Effect_bias
        ctrl_out = self.ctrl_Dense(S_tr_emb)
        out = treated_out*treated_mask + ctrl_out*ctrl_mask

        # out = tf.matmul(S_tr_emb, self.Effect_weights) + self.Effect_bias
        return out

    def __call__(self, inputs, **kwargs):
        tr = tf.expand_dims(inputs[:, 0], -1)
        s2c_ids = tf.cast(inputs[:, 1], dtype=tf.int32)  # [batch_size]
        cluster_rel = inputs[:, 2:902]
        loc = inputs[:, 902:904]
        his_data = tf.expand_dims(inputs[:, 904:-1], -1)

        # cluster embed
        cluster_rel = tf.reshape(cluster_rel, [-1, self.cluster_dim, self.cluster_dim])
        cl_emb = self.c_gcn_layer(cluster_rel)
        cfs_emb = tf.nn.embedding_lookup(cl_emb, s2c_ids)
        Rout_c = self.C_fc(cfs_emb)

        loc_emb = tf.nn.l2_normalize(self.loc_Dense(loc), axis=-1)
        his_emb = tf.nn.l2_normalize(self.RNN_layer(his_data), axis=-1)

        Rout_x = tf.concat([loc_emb, his_emb], axis=-1)

        z_c = self.VAE_layer.encoder_c(Rout_c)
        z_x = self.VAE_layer.encodet_x(Rout_x)

        z = tf.concat([z_c, z_x], axis=-1)

        s_tr = tf.concat([z,tr],axis=-1)
        outputs = self.predict_Out(s_tr,tr)
        return outputs,self.tr_Effect_weights