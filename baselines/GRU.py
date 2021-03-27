import tensorflow as tf
import tensorflow.keras.layers as layers

# use historical data and its location information to predict
# without community information

class GRU(tf.keras.Model):
    def __init__(self, rnn_dim):
        super(GRU,self).__init__()

        self.RNN_LAYER = layers.GRU(rnn_dim)
        self.loc_Dense = layers.Dense(2)
        self.out_layer = layers.Dense(1)

        self.tr_Effect_weights = self.add_weight(shape=[ rnn_dim + 3, 1], name='treated effect weights',
                                                 initializer=tf.keras.initializers.glorot_normal(), trainable=True)
        self.tr_Effect_bias = self.add_weight(shape=[1], name='treated effect bias', initializer=tf.keras.initializers.glorot_normal(),
                                              trainable=True)
        self.ctrl_Effect_weights = self.add_weight(shape=[rnn_dim + 3, 1], name='control effect weights',
                                                   initializer=tf.keras.initializers.glorot_normal(), trainable=True)
        self.ctrl_Effect_bias = self.add_weight(shape=[1], name='control effect bias', initializer=tf.keras.initializers.glorot_normal(),
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

    def call(self, inputs,**kwargs):
        tr = tf.expand_dims(inputs[:, 0], -1)
        loc = inputs[:, 902:904]
        his_data = tf.expand_dims(inputs[:, 904:-1], -1)

        his_emb = tf.nn.l2_normalize(self.RNN_LAYER(his_data),axis=-1)
        loc_emb = tf.nn.l2_normalize(self.loc_Dense(loc),axis=--1)

        com_emb = tf.concat([loc_emb,his_emb],axis=-1)
        com_emb = tf.concat([com_emb,tr],axis=-1)
        out = self.predict_Out(com_emb,tr)

        return out,self.tr_Effect_weights