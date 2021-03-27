import tensorflow as tf

def dot(A, B, sparse):
    if sparse:
        dot_res = tf.sparse.sparse_dense_matmul(A, B)
    else:
        dot_res = tf.matmul(A, B)
    return dot_res

class GCN_layer(tf.keras.layers.Layer):
    def __init__(self, hid_dim_list,  layer_num, activation=None,featureless=True):
        # hidden_dim_list: input and output dimension for each GCN layer
        # featureless: True(no feature matrix) default = True
        # hid_dim_list: size = layer_num+1,  hid[0]=feat_dim, hid[-1]=gcn_output
        super(GCN_layer, self).__init__()
        self.weightlist = []
        self.layer_num = layer_num
        for i in range(self.layer_num):
            self.weightlist.append(
                self.add_weight('weight_gcn' + str(i), [hid_dim_list[i], hid_dim_list[i + 1]]))
        self.featureless = featureless
        self.activation = activation

    def call(self, inputs, **kwargs):
        # for one timestamp
        # inputs: [adj,x]
        H_n = []
        for i in range(self.layer_num):
            if i == 0:
                if self.featureless:
                    h = self.weightlist[i]
                else:
                    h = dot(inputs[1], self.weightlist[i], sparse=False)
            else:
                # H(t) = H(t-1)*W[t]
                h = dot(H_n[-1], self.weightlist[i], sparse=False)
                H_n = []
            H_n.append(dot(inputs[0], h, sparse=False))
        if self.activation ==None:
            return H_n[-1]
        else:
            return self.activation(H_n[-1])
