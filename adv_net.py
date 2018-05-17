"""
Definition of adversarial examples generative network

"""
import tensorflow as tf

def adv_net(input_images):
    with tf.variable_scope('adv_encoder') as scope:
        width = 32
        height = 32
        batch_size = 128
        code_length = 6000

        input_images = input_images/255

        mean, var = tf.nn.moments(input_images, axes=tuple(range(1,len(input_images.shape))), keep_dims=True)
        normed_input_images = (input_images-mean)/var


        # Convolutional layer 1
        conv1 = tf.layers.conv2d(inputs=normed_input_images,
                                 filters=64,
                                 kernel_size=(3, 3),
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 activation=tf.nn.relu)

        # Convolutional output (flattened)
        conv_output = tf.contrib.layers.flatten(conv1)

        # Code layer
        code_layer = tf.layers.dense(inputs=conv_output,
                                     units=code_length,
                                     activation=tf.nn.relu)
        
        # Code output layer
        code_output = tf.layers.dense(inputs=code_layer,
                                      units=(height - 2) * (width - 2) * 3,
                                      activation=tf.nn.relu)

        # Deconvolution input
        deconv_input = tf.reshape(code_output, (batch_size, height - 2, width - 2, 3))

        # Deconvolution layer 1
        deconv1 = tf.layers.conv2d_transpose(inputs=deconv_input,
                                             filters=3,
                                             kernel_size=(3, 3),
                                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                             activation=tf.tanh)

        # renormed_deconv1 = deconv1*var+mean
        # Output batch
        output_images = input_images + deconv1
        output_images = tf.cast(tf.cast(tf.reshape(output_images, 
                                           (batch_size, height, width, 3)) * 255.0, tf.uint8), tf.float32)

        # Reconstruction L2 loss
        loss = tf.nn.l2_loss(deconv1, name='l2_loss')
    return loss, output_images


def adv_net_mask(input_images):
    with tf.variable_scope('adv_encoder') as scope:
        width = 32
        height = 32
        batch_size = 128
        # code_length = 6000

        input_images = input_images/255

        # arctan_images = tf.atan(((input_images*2)-1)*0.999999)

        # mean, var = tf.nn.moments(arctan_images, axes=tuple(range(1,len(input_images.shape))), keep_dims=True)
        # normed_input_images = (arctan_images-mean)/var

        mean, var = tf.nn.moments(input_images, axes=tuple(range(1,len(input_images.shape))), keep_dims=True)
        normed_input_images = (input_images-mean)/var

        # Convolutional layer 1
        conv1 = tf.layers.conv2d(inputs=normed_input_images,
                                 filters=64,
                                 kernel_size=(5, 5),
                                 # kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 activation=tf.nn.leaky_relu,
                                 padding='SAME',
                                 name='adv_conv1')

        # maxpool layer1
        maxpool1 = tf.layers.max_pooling2d(conv1, (3,3), (2,2), 'SAME')
        
        # Convolutional layer 2
        conv2 = tf.layers.conv2d(inputs=maxpool1,
                                 filters=128,
                                 kernel_size=(5, 5),
                                 # kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 activation=tf.nn.leaky_relu,
                                 padding='SAME',
                                 name='adv_conv2')

        # maxpool layer2
        maxpool2 = tf.layers.max_pooling2d(conv2, (3,3), (2,2), 'SAME')

        deconv1 = tf.layers.conv2d_transpose(maxpool2, 64, (5,5), (2,2), 'SAME',
                                             activation=tf.nn.leaky_relu,
                                             name='adv_deconv1')

        adv_mask = tf.layers.conv2d_transpose(deconv1, 3, (5,5), (2,2), 'SAME',
                                             activation=tf.nn.tanh,
                                             name='adv_deconv2')


        clip_norm = 1.5
        scaled_adv_mask = tf.clip_by_norm(adv_mask, clip_norm, axes=list(range(1,len(adv_mask.shape))))
        adv_images = tf.clip_by_value(scaled_adv_mask+input_images,0,1)
        output_images = tf.reshape(adv_images, (batch_size, height, width, 3)) * 255.0

        dif = adv_images - input_images

        # Display the training images in the visualizer.
        tf.summary.image('adv_images', output_images)

        # Reconstruction L2 loss
        mean_square_error = tf.reduce_mean(tf.square(dif), axis=list(range(1,len(dif.shape))))
        loss = tf.reduce_mean(mean_square_error, name='dis_loss')
    return loss, output_images

def adv_train_net(input_images, clip_norm=1.5):
    with tf.variable_scope('adv_encoder') as scope:
        width = 32
        height = 32
        batch_size = 128
        # code_length = 6000

        input_images = input_images/255

        # clip bound box
        mean, var = tf.nn.moments(input_images, axes=tuple(range(1,len(input_images.shape))), keep_dims=True)
        normed_input_images = (input_images-mean)/var

        # Convolutional layer 1
        conv1 = tf.layers.conv2d(inputs=normed_input_images,
                                 filters=64,
                                 kernel_size=(5, 5),
                                 # kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 activation=tf.nn.leaky_relu,
                                 padding='SAME',
                                 name='adv_conv1')

        # maxpool layer1
        maxpool1 = tf.layers.max_pooling2d(conv1, (3,3), (2,2), 'SAME')
        
        # Convolutional layer 2
        conv2 = tf.layers.conv2d(inputs=maxpool1,
                                 filters=128,
                                 kernel_size=(5, 5),
                                 # kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 activation=tf.nn.leaky_relu,
                                 padding='SAME',
                                 name='adv_conv2')

        # maxpool layer2
        maxpool2 = tf.layers.max_pooling2d(conv2, (3,3), (2,2), 'SAME')

        deconv1 = tf.layers.conv2d_transpose(maxpool2, 64, (5,5), (2,2), 'SAME',
                                             activation=tf.nn.leaky_relu,
                                             name='adv_deconv1')

        adv_mask = tf.layers.conv2d_transpose(deconv1, 3, (5,5), (2,2), 'SAME',
                                             activation=tf.nn.tanh,
                                             name='adv_deconv2')

        # clip bound box
        scaled_adv_mask = tf.clip_by_norm(adv_mask, clip_norm, axes=list(range(1,len(adv_mask.shape))))
        adv_images = tf.clip_by_value(scaled_adv_mask+input_images,0,1)
        output_images = tf.reshape(adv_images, (batch_size, height, width, 3)) * 255.0
        

        # output_images = tf.round(output_images)

        dif = adv_images - input_images

        # Display the training images in the visualizer.
        tf.summary.image('adv_images', output_images)

        # Reconstruction L2 loss
        mean_square_error = tf.reduce_mean(tf.square(dif), axis=list(range(1,len(dif.shape))))
        loss = tf.reduce_mean(mean_square_error, name='dis_loss')
    return loss, output_images

def adv_target_net(input_images, clip_norm=1.5):
    with tf.variable_scope('adv_encoder') as scope:
        width = 32
        height = 32
        batch_size = 128
        # code_length = 6000

        input_images = input_images/255

        # clip bound box
        mean, var = tf.nn.moments(input_images, axes=tuple(range(1,len(input_images.shape))), keep_dims=True)
        normed_input_images = (input_images-mean)/var

        # Convolutional layer 1
        conv1 = tf.layers.conv2d(inputs=normed_input_images,
                                 filters=64,
                                 kernel_size=(5, 5),
                                 # kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 activation=tf.nn.leaky_relu,
                                 padding='SAME',
                                 name='adv_conv1')

        # maxpool layer1
        maxpool1 = tf.layers.max_pooling2d(conv1, (3,3), (2,2), 'SAME')
        
        # Convolutional layer 2
        conv2 = tf.layers.conv2d(inputs=maxpool1,
                                 filters=128,
                                 kernel_size=(5, 5),
                                 # kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 activation=tf.nn.leaky_relu,
                                 padding='SAME',
                                 name='adv_conv2')

        # maxpool layer2
        maxpool2 = tf.layers.max_pooling2d(conv2, (3,3), (2,2), 'SAME')

        deconv1 = tf.layers.conv2d_transpose(maxpool2, 64, (5,5), (2,2), 'SAME',
                                             activation=tf.nn.leaky_relu,
                                             name='adv_deconv1')

        adv_mask = tf.layers.conv2d_transpose(deconv1, 3, (5,5), (2,2), 'SAME',
                                             activation=tf.nn.tanh,
                                             name='adv_deconv2')

        scaled_adv_mask = tf.clip_by_norm(adv_mask, clip_norm, axes=list(range(1,len(adv_mask.shape))))
        adv_images = tf.clip_by_value(scaled_adv_mask+input_images,0,1)
        output_images = tf.reshape(adv_images, (batch_size, height, width, 3)) * 255.0
        

        dif = adv_images - input_images

        tf.summary.image('adv_images', output_images)

        # Reconstruction L2 loss
        mean_square_error = tf.reduce_mean(tf.square(dif), axis=list(range(1,len(dif.shape))))
        loss = tf.reduce_mean(mean_square_error, name='dis_loss')
        
    return loss, output_images

def adv_target_net2(input_images, clip_norm=1.5):
    with tf.variable_scope('adv_encoder') as scope:
        width = 32
        height = 32
        batch_size = 128
        # code_length = 6000

        input_images = input_images/255

        # clip bound box
        mean, var = tf.nn.moments(input_images, axes=tuple(range(1,len(input_images.shape))), keep_dims=True)
        normed_input_images = (input_images-mean)/var

        # Convolutional layer 1
        conv1 = tf.layers.conv2d(inputs=normed_input_images,
                                 filters=32,
                                 kernel_size=(5, 5),
                                 # kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 activation=tf.nn.leaky_relu,
                                 padding='SAME',
                                 name='adv_conv1')

        # maxpool layer1
        maxpool1 = tf.layers.max_pooling2d(conv1, (3,3), (2,2), 'SAME')
        
        # Convolutional layer 2
        conv2 = tf.layers.conv2d(inputs=maxpool1,
                                 filters=64,
                                 kernel_size=(5, 5),
                                 # kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 activation=tf.nn.leaky_relu,
                                 padding='SAME',
                                 name='adv_conv2')

        # maxpool layer2
        maxpool2 = tf.layers.max_pooling2d(conv2, (3,3), (2,2), 'SAME')

        deconv1 = tf.layers.conv2d_transpose(maxpool2, 32, (5,5), (2,2), 'SAME',
                                             activation=tf.nn.leaky_relu,
                                             name='adv_deconv1')

        adv_mask = tf.layers.conv2d_transpose(deconv1, 3, (5,5), (2,2), 'SAME',
                                             activation=tf.nn.tanh,
                                             name='adv_deconv2')

        scaled_adv_mask = tf.clip_by_norm(adv_mask, clip_norm, axes=list(range(1,len(adv_mask.shape))))
        adv_images = tf.clip_by_value(scaled_adv_mask+input_images,0,1)
        output_images = tf.reshape(adv_images, (batch_size, height, width, 3)) * 255.0
        

        dif = adv_images - input_images

        tf.summary.image('adv_images', output_images)

        # Reconstruction L2 loss
        mean_square_error = tf.reduce_mean(tf.square(dif), axis=list(range(1,len(dif.shape))))
        loss = tf.reduce_mean(mean_square_error, name='dis_loss')
        
    return loss, output_images

def adv_target_net3(input_images, clip_norm=1.5):
    with tf.variable_scope('adv_encoder') as scope:
        width = 32
        height = 32
        batch_size = 128
        # code_length = 6000

        input_images = input_images/255

        # clip bound box
        mean, var = tf.nn.moments(input_images, axes=tuple(range(1,len(input_images.shape))), keep_dims=True)
        normed_input_images = (input_images-mean)/var

        # Convolutional layer 1
        conv1 = tf.layers.conv2d(inputs=normed_input_images,
                                 filters=64,
                                 kernel_size=(5, 5),
                                 # kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 activation=tf.nn.leaky_relu,
                                 padding='SAME',
                                 name='adv_conv1')

        # maxpool layer1
        maxpool1 = tf.layers.max_pooling2d(conv1, (3,3), (2,2), 'SAME')
        

        adv_mask = tf.layers.conv2d_transpose(maxpool1, 3, (5,5), (2,2), 'SAME',
                                             activation=tf.nn.tanh,
                                             name='adv_deconv2')

        scaled_adv_mask = tf.clip_by_norm(adv_mask, clip_norm, axes=list(range(1,len(adv_mask.shape))))
        adv_images = tf.clip_by_value(scaled_adv_mask+input_images,0,1)
        output_images = tf.reshape(adv_images, (batch_size, height, width, 3)) * 255.0
        

        dif = adv_images - input_images

        tf.summary.image('adv_images', output_images)

        # Reconstruction L2 loss
        mean_square_error = tf.reduce_mean(tf.square(dif), axis=list(range(1,len(dif.shape))))
        loss = tf.reduce_mean(mean_square_error, name='dis_loss')
        
    return loss, output_images


def adv_train_arctan_net(input_images, clip_norm=1.5):
    with tf.variable_scope('adv_encoder') as scope:
        width = 32
        height = 32
        batch_size = 128
        # code_length = 6000

        input_images = input_images/255

        arctan_images = tf.atanh(((input_images*2)-1)*0.999999)

        mean, var = tf.nn.moments(arctan_images, axes=tuple(range(1,len(input_images.shape))), keep_dims=True)
        normed_input_images = (arctan_images-mean)/var

        # Convolutional layer 1
        conv1 = tf.layers.conv2d(inputs=normed_input_images,
                                 filters=64,
                                 kernel_size=(5, 5),
                                 # kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 activation=tf.nn.leaky_relu,
                                 padding='SAME',
                                 name='adv_conv1')

        # maxpool layer1
        maxpool1 = tf.layers.max_pooling2d(conv1, (3,3), (2,2), 'SAME')
        
        # Convolutional layer 2
        conv2 = tf.layers.conv2d(inputs=maxpool1,
                                 filters=128,
                                 kernel_size=(5, 5),
                                 # kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 activation=tf.nn.leaky_relu,
                                 padding='SAME',
                                 name='adv_conv2')

        # maxpool layer2
        maxpool2 = tf.layers.max_pooling2d(conv2, (3,3), (2,2), 'SAME')

        deconv1 = tf.layers.conv2d_transpose(maxpool2, 64, (5,5), (2,2), 'SAME',
                                             activation=tf.nn.leaky_relu,
                                             name='adv_deconv1')

        adv_mask = tf.layers.conv2d_transpose(deconv1, 3, (5,5), (2,2), 'SAME',
                                             # activation=tf.nn.tanh,
                                             name='adv_deconv2')

        arctan_adv_images = adv_mask + normed_input_images
        unscaled_adv_images = tf.tanh(arctan_adv_images)
        unscaled_diff = unscaled_adv_images - input_images
        # clip_norm = 1.5
        scaled_dif = tf.clip_by_norm(unscaled_diff, clip_norm)
        adv_images = tf.clip_by_value(scaled_dif+input_images,0,1)
        output_images = tf.reshape(adv_images, (batch_size, height, width, 3)) * 255.0

        dif = adv_images - input_images

        # Display the training images in the visualizer.
        tf.summary.image('adv_images', output_images)

        # Reconstruction L2 loss
        mean_square_error = tf.reduce_mean(tf.square(dif), axis=list(range(1,len(dif.shape))))
        loss = tf.reduce_mean(mean_square_error, name='dis_loss')
        
    return loss, output_images