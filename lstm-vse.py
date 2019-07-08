import tensorflow as tf
import numpy as np
import json
import random
import vgg16

data_path = './files/'

with open(data_path+'vocabulary.json') as f:
    word_dict = json.load(f)

def get_variable(type, shape, name, reuse = False):
    if type == 'W':
        with tf.variable_scope('model', reuse=reuse):
            var = tf.get_variable(name=name, shape=shape, dtype=tf.float32,
                                  initializer=tf.random_uniform_initializer(minval=-0.08,maxval=0.08))
        tf.add_to_collection('regular_losses', tf.contrib.layers.l2_regularizer(0.005)(var))
        return var
    elif type == 'b':
        with tf.variable_scope('model', reuse=reuse):
            var = tf.get_variable(name=name, shape=shape, dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
        return var


class lstm_vse:
    def __init__(self):
        self.imageFeaDim = 4096
        self.wordDictDim = len(word_dict)
        self.embeddingDim = 512
        self.max_len = 8
        self.m = 0.2
        self.batch_size = None

        self.word_embedding = None
        self.img_embedding = None
        self.rnn_img_embedding = None
        self.img_feature = None
        self.test_feature = None
        self.item_title = None
        self.sequence_length = None
        self.label = None
        self.mode = 'training'

        self.imgFea_dict = {}
        self.textFea_dict = {}

    def bi_lstm(self, img_feature, sequence_length):
        '''
        :param img_featre:  [None, feature_dim]
        :param sequence_length: list
        :return:
        '''
        img_feature = tf.nn.l2_normalize(img_feature, dim=1)

        imgFnn_W = get_variable(type='W', shape=[self.imageFeaDim, self.embeddingDim], mean=0, stddev=0.01,
                                name='imgFnn_W')
        imgFnn_b = get_variable(type='b', shape=[self.embeddingDim], mean=0, stddev=0.01, name='imgFnn_b')

        img_feature = tf.matmul(img_feature, imgFnn_W) + imgFnn_b

        self.sequence_length = sequence_length
        img_feature = tf.split(img_feature, self.sequence_length)
        for batch in img_feature:
            batch = tf.transpose(batch)
            batch = tf.pad(batch, [[0,0], [0, self.max_len-tf.shape(batch)[1]]])
            batch = tf.transpose(batch)

            if self.rnn_img_embedding == None:
                self.rnn_img_embedding = batch
            else:
                self.rnn_img_embedding = tf.concat([self.rnn_img_embedding, batch], axis=0)
        self.rnn_img_embedding = tf.reshape(self.rnn_img_embedding, shape=[-1, self.max_len, self.embeddingDim])

        # Forward LSTM.
        f_lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.embeddingDim, state_is_tuple=True)
        # Backward LSTM.
        b_lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.embeddingDim, state_is_tuple=True)

        self.lstm_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=f_lstm_cell,
                                                                              cell_bw=b_lstm_cell,
                                                                              inputs=self.rnn_img_embedding,
                                                                              dtype=tf.float32,
                                                                              initial_state_fw=None,
                                                                              initial_state_bw=None,
                                                                              sequence_length=self.sequence_length
                                                                              )
        ### training
        fw_lstm_outputs = self.lstm_outputs[0]
        bw_lstm_outputs = self.lstm_outputs[1]

        ## hidden_output
        fw_hidden = tf.reshape(fw_lstm_outputs[:, :-1, :], shape=[-1, self.embeddingDim])
        bw_hidden = tf.reshape(bw_lstm_outputs[:, :-1, :], shape=[-1, self.embeddingDim])

        ## target
        fw_target = tf.reshape(self.rnn_img_embedding[:, 1:, :], shape=[-1, self.embeddingDim])
        bw_target = tf.reverse_sequence(self.rnn_img_embedding,seq_lengths=sequence_length, seq_dim=1, batch_dim=0)
        bw_target = tf.reshape(bw_target[:, 1:, :], shape=[-1, self.embeddingDim])

        ##
        x_img = tf.reshape(self.rnn_img_embedding, shape=[-1, self.embeddingDim])  #[None, embed_size]

        fw_ht = tf.reduce_sum(tf.multiply(fw_hidden, fw_target), axis=1)
        bw_ht = tf.reduce_sum(tf.multiply(bw_hidden, bw_target), axis=1)
        mask = tf.cast(fw_ht, tf.bool)

        fw_hx = tf.matmul(fw_hidden, tf.transpose(x_img))
        bw_hx = tf.matmul(bw_hidden, tf.transpose(x_img))


        fw_Pr = tf.divide(tf.exp(tf.boolean_mask(fw_ht, mask)),
                          tf.reduce_sum(tf.exp(tf.boolean_mask(fw_hx, mask)), axis=1))
        bw_Pr = tf.divide(tf.exp(tf.boolean_mask(bw_ht, mask)),
                          tf.reduce_sum(tf.exp(tf.boolean_mask(bw_hx, mask)), axis=1))

        self.lstm_loss = - tf.reduce_mean(tf.log(fw_Pr)) - tf.reduce_mean(tf.log(bw_Pr))

        ## testing
        test_fw_loss = tf.reduce_sum(tf.multiply(fw_hidden, fw_target), axis=1)
        test_fw_loss = tf.reduce_sum(tf.reshape(test_fw_loss, shape=[-1, self.max_len-1]), axis=1)
        test_bw_loss = tf.reduce_sum(tf.multiply(bw_hidden, bw_target), axis=1)
        test_bw_loss = tf.reduce_sum(tf.reshape(test_bw_loss, shape=[-1, self.max_len - 1]), axis=1)

        self.score = (test_fw_loss+test_bw_loss) / tf.cast(self.sequence_length, tf.float32)

        self.accuracy =  tf.nn.top_k(self.score, k=int(self.batch_size/2)).indices

    def build_textFea_dict(self, text_path):
        with open(text_path, 'r') as f:
            data = json.load(f)
        for i in data:
            for ii in i['items']:
                item = '%s/%s.jpg'%(i['set_id'], ii['index'])
                name = ii['name'].split(' ')
                wordFea = [0.]*self.wordDictDim
                wordFea = np.array(wordFea).astype(np.float32)
                for word in name:
                    wordFea[word_dict.index(word)] = 1 / len(name)
                if item not in self.textFea_dict.keys():
                    self.textFea_dict[item] = wordFea

    def vse(self, item_title, img_feature):
        ## word
        wordEmbMatrix = get_variable(type='W', shape=[self.wordDictDim, self.embeddingDim], mean=0, stddev=0.01, name='textEmb')

        self.word_embedding = tf.matmul(item_title, wordEmbMatrix)
        self.word_embedding = tf.nn.l2_normalize(self.word_embedding, dim=1)

        ## image
        imageEmb_W = get_variable(type='W', shape=[self.imageFeaDim, self.embeddingDim], mean=0, stddev=0.01, name='imageEmb')
        img_feature = tf.nn.l2_normalize(img_feature, dim=1)
        self.img_embedding = tf.matmul(img_feature, imageEmb_W)

        matrix = tf.matmul(self.img_embedding, tf.transpose(self.word_embedding),name='matrix')
        fv = matrix-tf.transpose(tf.diag_part(matrix)) + self.m
        vf = tf.transpose(matrix)-tf.transpose(tf.diag_part(matrix)) + self.m

        self.vse_loss = tf.reduce_sum(tf.maximum(tf.zeros(shape=tf.shape(fv)), fv)) \
                    + tf.reduce_sum(tf.maximum(tf.zeros(shape=tf.shape(vf)), vf))


    def train(self):
        self.img_feature = tf.placeholder(dtype=tf.float32, shape=[None, self.imageFeaDim], name='image')
        self.item_title = tf.placeholder(dtype=tf.float32, shape=[None, self.wordDictDim], name='text')
        self.sequence_length = tf.placeholder(dtype=tf.int32, shape=[self.batch_size], name='sequence_length')

        self.vse(item_title=self.item_title, img_feature=self.img_feature)
        self.build_textFea_dict(data_path+'train_no_dup.json')

        self.bi_lstm(img_feature=self.img_feature, sequence_length=self.sequence_length)
        self.loss = self.lstm_loss + 0.1*self.vse_loss

        self.global_step = tf.Variable(0, dtype=tf.int64, name='global_step', trainable=False)
        learning_rate = tf.train.exponential_decay(0.001, self.global_step, decay_steps=100, decay_rate=0.90,staircase=False)
        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

