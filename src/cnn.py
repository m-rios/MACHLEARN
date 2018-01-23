import tensorflow as tf

class CNN( object ):
    def __init__(self):
        # Convolutional Layer 1 
        self.filter_size1 = 2 
        self.num_filters1 = 16

        # Convolutional Layer 2 
        self.filter_size2 = 2 
        self.num_filters2 = 32

        # fully connected layer 
        self.fc1 = 192

        # fully connected layer 
        self.fc2 = 96

        # number of channels. Since input is 64*4
        self.num_channels = 4

        # dimensions of the input 
        self.input = 64*4 + 2 

        classes = ['Win', 'Loss','Draw']
        self.out = len(classes)

        self.X = tf.placeholder("float", shape=[None, self.input])

        self.X_in = tf.gather(self.X, [i for i in range(64*4)], axis = 1)
        self.turn = tf.gather(self.X, [64*4, 64*4+1], axis = 1)

        self.X_input = tf.reshape(self.X_in, [-1, 8, 8, self.num_channels])

        self.layer_conv1 = self.new_conv_layer(input=self.X_input , num_input_channels= self.num_channels, 
            filter_size=self.filter_size1, num_filters= self.num_filters1, use_pooling=True)
        
        self.layer_conv2 = self.new_conv_layer(input=self.layer_conv1 , num_input_channels= self.num_filters1, 
            filter_size=self.filter_size2, num_filters= self.num_filters2, use_pooling=True)

        self.layer_flat, self.num_features = self.flatten_layer(self.layer_conv2)
        
        self.layer_flat = tf.concat([self.layer_flat, self.turn],1)
        self.num_features = self.num_features + 2 

        self.layer_fc1 = self.new_fc_layer(input=self.layer_flat, num_inputs=self.num_features, num_outputs=self.fc1, use_relu=True)
        self.layer_fc2 = self.new_fc_layer(input=self.layer_fc1, num_inputs=self.fc1, num_outputs=self.fc2, use_relu=True)
        self.last_layer = self.new_fc_layer(input=self.layer_fc2, num_inputs=self.fc2, num_outputs=self.out,use_relu=False)
        self.class_prob = tf.nn.softmax(self.last_layer)
        self.ev = tf.argmax(self.class_prob, dimension=1)

        
        # self.ev = tf.sign(tf.subtract(tf.nn.softmax(self.layer_fc2), tf.constant(0.5)))
        self.trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    
    def new_weights(self, shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

         # biases are fixed, so we only train weights 
    def new_biases(self, length):
        return tf.Variable(tf.constant(0.05, shape=[length]))   


    def new_conv_layer(self, input, num_input_channels, filter_size, num_filters, use_pooling=True):
        # num_input_channels = num_filters in last layer 

        #shape of filter weights for convolution 
        shape = [filter_size, filter_size, num_input_channels, num_filters]

        #create new weights with the given shape
        weights = self.new_weights(shape=shape)

        #new biases, one for each filter 
        biases = self.new_biases(length=num_filters)

        # the CNN layer, padding = "same" means the input is padded with zero's. We have strides of 1,1,1,1
        layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')

        # a bias value added to each filter channel
        layer += biases

        # It calculates max(x, 0) for each input value x. This adds some non-linearity.
        layer = tf.nn.relu(layer)

        #  This is 2x2 max-pooling, which means that we consider 2x2 windows and select the largest value 
        # in each window. Then we move 2 pixels to the next window.
        if use_pooling:
            layer = tf.nn.max_pool(value=layer,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        return layer


    # to flatten the layer of the fully connected part of the CNN
    def flatten_layer(self, layer):

        # Get the shape of the input layer.
        layer_shape = layer.get_shape()

        # The shape of the flattened layer should be [num_images, img_height * img_width * num_channels]
        num_features = layer_shape[1:4].num_elements()
        layer_flat = tf.reshape(layer, [-1, num_features])

        return layer_flat, num_features

    # defining the fully-connected layer 
    def new_fc_layer(self, input, num_inputs, num_outputs, use_relu=True):
        # Create new weights and biases.
        weights = self.new_weights(shape=[num_inputs, num_outputs])
        biases = self.new_biases(length=num_outputs)

        # Calculate the layer as the matrix multiplication of
        # the input and weights, and then add the bias-values.
        layer = tf.matmul(input, weights) + biases

        # Use ReLU?
        if use_relu:
            layer = tf.nn.relu(layer)

        return layer