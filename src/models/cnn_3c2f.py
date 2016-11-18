import tensorflow as tf
from models.cnn import CNN
from modules.debug import debug, error

class CNN3C2F(CNN):
    
    # initialize tensorflow variables which are required to learning
    def initialize(self):
        with tf.device(self.flag["TRAIN_DEVICE_ID"]):    

            # These placeholder nodes will be fed a batch of training data at each
            # training step using the {feed_dict} argument to the Run() call below.

            self.train_data_node = tf.placeholder(
                tf.float32,
                shape=(self.hp["BATCH_SIZE"], self.data["IMAGE_SIZE"], self.data["IMAGE_SIZE"], self.data["NUM_CHANNELS"]))

            self.train_labels_node = tf.placeholder(tf.int64, shape=(self.hp["BATCH_SIZE"],))

            self.eval_data = tf.placeholder(
                tf.float32,
                shape=(self.hp["EVAL_BATCH_SIZE"], self.data["IMAGE_SIZE"], self.data["IMAGE_SIZE"], self.data["NUM_CHANNELS"]))

            # The variables below hold all the trainable weights. They are passed an initial value which will be assigned when we call:
            # {tf.initialize_all_variables().run()}
            self.conv1_weights = tf.Variable(
                    tf.truncated_normal([self.hp["FILTER_SIZE"], self.hp["FILTER_SIZE"], self.data["NUM_CHANNELS"], self.hp["CONV1_DEPTH"]],
                                        stddev=self.hp["INIT_STDDEV"],
                                        seed=self.hp["SEED"]))

            self.conv1_biases = tf.Variable(tf.zeros([self.hp["CONV1_DEPTH"]]))

            self.conv2_weights = tf.Variable(
                    tf.truncated_normal([self.hp["FILTER_SIZE"], self.hp["FILTER_SIZE"], self.hp["CONV1_DEPTH"], self.hp["CONV2_DEPTH"]],
                                        stddev=self.hp["INIT_STDDEV"],
                                        seed=self.hp["SEED"]))
            self.conv2_biases = tf.Variable(tf.constant(self.hp["INIT_WEIGHT_VALUE"], shape=[self.hp["CONV2_DEPTH"]]))

            self.conv3_weights = tf.Variable(
                    tf.truncated_normal([self.hp["FILTER_SIZE"], self.hp["FILTER_SIZE"], self.hp["CONV2_DEPTH"], self.hp["CONV3_DEPTH"]],
                                        stddev=self.hp["INIT_STDDEV"],
                                        seed=self.hp["SEED"]))
            self.conv3_biases = tf.Variable(tf.constant(self.hp["INIT_WEIGHT_VALUE"], shape=[self.hp["CONV3_DEPTH"]]))            
            
            resized = int(round(self.data["IMAGE_SIZE"] / self.hp["STRIDE_SIZE"] / self.hp["STRIDE_SIZE"] / self.hp["STRIDE_SIZE"]))
            self.fc1_weights = tf.Variable(    # fully connected
                    tf.truncated_normal(                          
                            [resized * resized * self.hp["CONV3_DEPTH"], 
                             self.hp["FC1_WIDTH"]],
                            stddev=self.hp["INIT_STDDEV"],
                            seed=self.hp["SEED"]))

            self.fc1_biases = tf.Variable(tf.constant(self.hp["INIT_WEIGHT_VALUE"], shape=[self.hp["FC1_WIDTH"]]))

            self.fc2_weights = tf.Variable(    # fully connected
                    tf.truncated_normal(                          
                            [self.hp["FC1_WIDTH"], 
                             self.hp["FC2_WIDTH"]],
                            stddev=self.hp["INIT_STDDEV"],
                            seed=self.hp["SEED"]))

            self.fc2_biases = tf.Variable(tf.constant(self.hp["INIT_WEIGHT_VALUE"], shape=[self.hp["FC2_WIDTH"]]))
            
            # output layer
            self.output_weights = tf.Variable(tf.truncated_normal([self.hp["FC2_WIDTH"], 
                                                              self.data["NUM_LABELS"]],
                                                            stddev=self.hp["INIT_STDDEV"],
                                                            seed=self.hp["SEED"]))
            self.output_biases = tf.Variable(tf.constant(self.hp["INIT_WEIGHT_VALUE"], shape=[self.data["NUM_LABELS"]]))    
            

    # We will replicate the model structure for the training subgraph, as well
    # as the evaluation subgraphs, while sharing the trainable parameters.
    def createModel(self, data, train=False):
        with tf.device(self.flag["TRAIN_DEVICE_ID"]):
            """The Model definition."""

            # 2D convolution, with 'SAME' padding (i.e. the output feature map has
            # the same size as the input). Note that {strides} is a 4D array whose
            # shape matches the data layout: [image index, y, x, depth].
            conv = tf.nn.conv2d(data,
                                self.conv1_weights,
                                strides=[1, 1, 1, 1],
                                padding='SAME')

            # Bias and rectified linear non-linearity.
            relu = tf.nn.relu(tf.nn.bias_add(conv, self.conv1_biases))

            # Max pooling. The kernel size spec {ksize} also follows the layout of
            # the data. Here we have a pooling window of 2, and a stride of 2.
            pool = tf.nn.max_pool(relu,
                                  ksize=[1, self.hp["POOLING_SIZE"], self.hp["POOLING_SIZE"], 1],
                                  strides=[1, self.hp["STRIDE_SIZE"], self.hp["STRIDE_SIZE"], 1],
                                  padding=self.hp["PADDING"])

            conv = tf.nn.conv2d(pool,
                                self.conv2_weights,
                                strides=[1, 1, 1, 1],
                                padding='SAME')

            relu = tf.nn.relu(tf.nn.bias_add(conv, self.conv2_biases))

            pool = tf.nn.max_pool(relu,
                                  ksize=[1, self.hp["POOLING_SIZE"], self.hp["POOLING_SIZE"], 1],
                                  strides=[1, self.hp["STRIDE_SIZE"], self.hp["STRIDE_SIZE"], 1],
                                  padding=self.hp["PADDING"])
            
            
            conv = tf.nn.conv2d(pool,
                                self.conv3_weights,
                                strides=[1, 1, 1, 1],
                                padding='SAME')

            relu = tf.nn.relu(tf.nn.bias_add(conv, self.conv3_biases))

            pool = tf.nn.max_pool(relu,
                                  ksize=[1, self.hp["POOLING_SIZE"], self.hp["POOLING_SIZE"], 1],
                                  strides=[1, self.hp["STRIDE_SIZE"], self.hp["STRIDE_SIZE"], 1],
                                  padding=self.hp["PADDING"])
            
            # Reshape the feature map cuboid into a 2D matrix to feed it to the
            # fully connected layers.
            pool_shape = pool.get_shape().as_list()
            debug("pool shape: "+ str(pool_shape))
            reshape = tf.reshape(pool,
                [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]]) # XXX:I couldn't understand what this operation did

            # Fully connected layer. Note that the '+' operation automatically broadcasts the biases.
            debug("reshaped: " + str(reshape.get_shape()))
            debug("fc1 shape: " + str(self.fc1_weights.get_shape()))
            

            hidden1 = tf.nn.relu(tf.matmul(reshape, self.fc1_weights) + self.fc1_biases)
            
            hidden2 = tf.nn.relu(tf.matmul(hidden1, self.fc2_weights) + self.fc2_biases)

            # Add a dropout during training only. Dropout also scales
            # activations such that no rescaling is needed at evaluation time.
            if train:
                hidden1 = tf.nn.dropout(hidden1, 1.0 - self.hp["DROPOUT_RATE"], seed=self.hp["SEED"])
                hidden2 = tf.nn.dropout(hidden2, 1.0 - self.hp["DROPOUT_RATE"], seed=self.hp["SEED"])

        return tf.matmul(hidden2, self.output_weights) + self.output_biases


if __name__ == '__main__':
    
    import mnist_data as mnist
    dataset = mnist.import_dataset()                                    
    
    from modules.hpmgr import HPVManager 
   
    hpv = HPVManager('HPV_002.ini', ini_dir='../config/')
    model = CNN3C2F(dataset)
    print model.learn(hpv)
