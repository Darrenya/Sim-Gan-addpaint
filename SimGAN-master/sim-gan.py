"""
Implementation of `3.1 Appearance-based Gaze Estimation` from
[Learning from Simulated and Unsupervised Images through Adversarial Training](https://arxiv.org/pdf/1612.07828v1.pdf).

Note: Only Python 3 support currently.
"""
from __future__ import print_function
import os
import sys

from keras import applications
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing import image
import h5py
import numpy as np
import tensorflow as tf
import scipy.io as scio 
from dlutils import plot_image_batch_w_labels##看github   加载的必须文件   作用百度

from utils.image_history_buffer import ImageHistoryBuffer


#
# directories
#

path = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(path, 'cache')##  文件位置
#sess = tf.Session()
#writer = tf.summary.FileWriter("D://logs",tf.get_default_graph())  

# path = os.path.dirname(os.path.abspath('.'))
# data_dir = os.path.join('..', 'input')
# cache_dir = '.'

# # load the data file and extract dimensions
# with scio.loadmat(os.path.join(data_dir,'day01.mat'),'r') as t_file:
    # print(list(t_file.keys()))
    # assert 'image' in t_file, "Images are missing"
    # assert 'look_vec' in t_file, "Look vector is missing"
    # assert 'path' in t_file, "Paths are missing"
    # print('Synthetic images found:',len(t_file['image']))
    # for _, (ikey, ival) in zip(range(1), t_file['image'].items()):
        # print('image',ikey,'shape:',ival.shape)
        # img_height, img_width = ival.shape
        # img_channels = 1
    # syn_image_stack = np.stack([np.expand_dims(a,-1) for a in t_file['image'].values()],0)

# with h5py.File(os.path.join(data_dir,'real_gaze.h5'),'r') as t_file:
    # print(list(t_file.keys()))
    # assert 'image' in t_file, "Images are missing"
    # print('Real Images found:',len(t_file['image']))
    # for _, (ikey, ival) in zip(range(1), t_file['image'].items()):
        # print('image',ikey,'shape:',ival.shape)
        # img_height, img_width = ival.shape
        # img_channels = 1
    # real_image_stack = np.stack([np.expand_dims(a,-1) for a in t_file['image'].values()],0)

#
# image dimensions
#

img_width = 55
img_height = 35
img_channels = 1##图像  如果不是图像呢？

#
# training params   训练参数
#

nb_steps = 1000
batch_size =5 ##Batch 的选择，首先决定的是下降的方向。如果数据集比较小，完全可以采用全数据集 
k_d = 1  # number of discriminator updates per step  每步的鉴别器更新次数
k_g = 2  # number of generative network updates per step  每步生成器更新次数
log_interval = 10  ##日志间隔x(训练间隔是对的)  分类


def refiner_network(input_image_tensor):#定义精炼网络
    """
    The refiner network, Rθ, is a residual network (ResNet). It modifies the synthetic image on a pixel level, rather
    than holistically modifying the image content, preserving the global structure and annotations.

    :param input_image_tensor: Input tensor that corresponds to a synthetic image.
    :return: Output tensor that corresponds to a refined synthetic image.
    """
    #with tf.name_scope('refiner_network'):
    def resnet_block(input_features, nb_features=64, nb_kernel_rows=3, nb_kernel_cols=3): ##卷积神经网络  
        #with tf.name_scope('resnet_block')：
        """
        A ResNet block with two `nb_kernel_rows` x `nb_kernel_cols` convolutional layers,
        each with `nb_features` feature maps.

        See Figure 6 in https://arxiv.org/pdf/1612.07828v1.pdf.

        :param input_features: Input tensor to ResNet block.
        :return: Output tensor from ResNet block.来自ResNet块的输出张量
        """
        with tf.name_scope('resnet_block'):
        y = layers.Convolution2D(nb_features, nb_kernel_rows, nb_kernel_cols, border_mode='same')(input_features)
        y = layers.Activation('relu')(y)
        y = layers.Convolution2D(nb_features, nb_kernel_rows, nb_kernel_cols, border_mode='same')(y)

        y = layers.merge([input_features, y], mode='sum')
        return layers.Activation('relu')(y)#relu：线性整流函数  激励函数

    # an input image of size w × h is convolved with 3 × 3 filters that output 64 feature maps
    x = layers.Convolution2D(64, 3, 3, border_mode='same', activation='relu')(input_image_tensor)

    # the output is passed through 4 ResNet blocks
    for _ in range(4):
        x = resnet_block(x)

    # the output of the last ResNet block is passed to a 1 × 1 convolutional layer producing 1 feature map
    # corresponding to the refined synthetic image
    return layers.Convolution2D(img_channels, 1, 1, border_mode='same', activation='tanh')(x)


def discriminator_network(input_image_tensor):#判别网络
#with tf.name_scope('discriminator_network')：
    """
    The discriminator network, Dφ, contains 5 convolution layers and 2 max-pooling layers.

    :param input_image_tensor: Input tensor corresponding to an image, either real or refined.
    :return: Output tensor that corresponds to the probability of whether an image is real or refined.
    """
    x = layers.Convolution2D(96, 3, 3, border_mode='same', subsample=(2, 2), activation='relu')(input_image_tensor)
    x = layers.Convolution2D(64, 3, 3, border_mode='same', subsample=(2, 2), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), border_mode='same', strides=(1, 1))(x)
    x = layers.Convolution2D(32, 3, 3, border_mode='same', subsample=(1, 1), activation='relu')(x)
    x = layers.Convolution2D(32, 1, 1, border_mode='same', subsample=(1, 1), activation='relu')(x)
    x = layers.Convolution2D(2, 1, 1, border_mode='same', subsample=(1, 1), activation='relu')(x)

    # here one feature map corresponds to `is_real` and the other to `is_refined`,
    # and the custom loss function is then `tf.nn.sparse_softmax_cross_entropy_with_logits`
    return layers.Reshape((-1, 2))(x)#返回精炼还是真实  


def adversarial_training(synthesis_eyes_dir, mpii_gaze_dir, refiner_model_path=None, discriminator_model_path=None):
    """Adversarial training of refiner network Rθ and discriminator network Dφ."""
    #
    # define model input and output tensors
    #
#with tf.name_scope('adversarial_training')：
    synthetic_image_tensor = layers.Input(shape=(img_height, img_width, img_channels))
    refined_image_tensor = refiner_network(synthetic_image_tensor)

    refined_or_real_image_tensor = layers.Input(shape=(img_height, img_width, img_channels))
    discriminator_output = discriminator_network(refined_or_real_image_tensor)

    #
    # define models
    #

    refiner_model = models.Model(input=synthetic_image_tensor, output=refined_image_tensor, name='refiner')
    discriminator_model = models.Model(input=refined_or_real_image_tensor, output=discriminator_output,
                                       name='discriminator')

    # combined must output the refined image along w/ the disc's classification of it for the refiner's self-reg loss
	#合并后必须沿着disc's的分类输出精炼图像，以确定refiner's的自调整损失
    refiner_model_output = refiner_model(synthetic_image_tensor)
    combined_output = discriminator_model(refiner_model_output)
    combined_model = models.Model(input=synthetic_image_tensor, output=[refiner_model_output, combined_output],
                                  name='combined')#合成输出

    discriminator_model_output_shape = discriminator_model.output_shape

    print(refiner_model.summary())
    print(discriminator_model.summary())
    print(combined_model.summary())

    #
    # define custom l1 loss function for the refiner
    #

    def self_regularization_loss(y_true, y_pred):
	#with tf.name_scope('self_regularization_loss')：
        delta = 0.0001  # FIXME: need to figure out an appropriate value for this
        return tf.multiply(delta, tf.reduce_sum(tf.abs(y_pred - y_true)))

    #
    # define custom local adversarial loss (softmax for each image section) for the discriminator
    # the adversarial loss function is the sum of the cross-entropy losses over the local patches
    #

    def local_adversarial_loss(y_true, y_pred):
	#with tf.name_scope('local_adversarial_loss')：
        # y_true and y_pred have shape (batch_size, # of local patches, 2), but really we just want to average over
        # the local patches and batch size so we can reshape to (batch_size * # of local patches, 2)
        y_true = tf.reshape(y_true, (-1, 2))
        y_pred = tf.reshape(y_pred, (-1, 2))
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)

        return tf.reduce_mean(loss)

    #
    # compile models
    #

    sgd = optimizers.SGD(lr=0.001)

    refiner_model.compile(optimizer=sgd, loss=self_regularization_loss)
    discriminator_model.compile(optimizer=sgd, loss=local_adversarial_loss)
    discriminator_model.trainable = False
    combined_model.compile(optimizer=sgd, loss=[self_regularization_loss, local_adversarial_loss])

    #
    # data generators
    #

    datagen = image.ImageDataGenerator(
        preprocessing_function=applications.xception.preprocess_input)

    flow_from_directory_params = {'target_size': (img_height, img_width),
                                  'color_mode': 'grayscale' if img_channels == 1 else 'rgb',
                                  'class_mode': None,
                                  'batch_size': batch_size}

    synthetic_generator = datagen.flow_from_directory(
        directory=synthesis_eyes_dir,
        **flow_from_directory_params
    )

    real_generator = datagen.flow_from_directory(
        directory=mpii_gaze_dir,
        **flow_from_directory_params
    )

    def get_image_batch(generator):
        """keras generators may generate an incomplete batch for the last batch"""
        img_batch = generator.next()
        if len(img_batch) != batch_size:
            img_batch = generator.next()

        assert len(img_batch) == batch_size

        return img_batch

    # the target labels for the cross-entropy loss layer are 0 for every yj (real) and 1 for every xi (refined)
    y_real = np.array([[[1.0, 0.0]] * discriminator_model_output_shape[1]] * batch_size)
    y_refined = np.array([[[0.0, 1.0]] * discriminator_model_output_shape[1]] * batch_size)
    assert y_real.shape == (batch_size, discriminator_model_output_shape[1], 2)

    if not refiner_model_path:
        # we first train the Rθ network with just self-regularization loss for 1,000 steps
        print('pre-training the refiner network...')
        gen_loss = np.zeros(shape=len(refiner_model.metrics_names))

        for i in range(1000):
            synthetic_image_batch = get_image_batch(synthetic_generator)
            gen_loss = np.add(refiner_model.train_on_batch(synthetic_image_batch, synthetic_image_batch), gen_loss)

            # log every `log_interval` steps
            if not i % log_interval:
                figure_name = 'refined_image_batch_pre_train_step_{}.png'.format(i)
                print('Saving batch of refined images during pre-training at step: {}.'.format(i))

                synthetic_image_batch = get_image_batch(synthetic_generator)
                plot_image_batch_w_labels.plot_batch(
                    np.concatenate((synthetic_image_batch, refiner_model.predict_on_batch(synthetic_image_batch))),
                    os.path.join(cache_dir, figure_name),
                    label_batch=['Synthetic'] * batch_size + ['Refined'] * batch_size)

                print('Refiner model self regularization loss: {}.'.format(gen_loss / log_interval))
                gen_loss = np.zeros(shape=len(refiner_model.metrics_names))

        refiner_model.save(os.path.join(cache_dir, 'refiner_model_pre_trained.h5'))
    else:
        refiner_model.load_weights(refiner_model_path)

    if not discriminator_model_path:
        # and Dφ for 200 steps (one mini-batch for refined images, another for real)
        print('pre-training the discriminator network...')
        disc_loss = np.zeros(shape=len(discriminator_model.metrics_names))

        for _ in range(100):
            real_image_batch = get_image_batch(real_generator)
            disc_loss = np.add(discriminator_model.train_on_batch(real_image_batch, y_real), disc_loss)

            synthetic_image_batch = get_image_batch(synthetic_generator)
            refined_image_batch = refiner_model.predict_on_batch(synthetic_image_batch)
            disc_loss = np.add(discriminator_model.train_on_batch(refined_image_batch, y_refined), disc_loss)

        discriminator_model.save(os.path.join(cache_dir, 'discriminator_model_pre_trained.h5'))

        # hard-coded for now
        print('Discriminator model loss: {}.'.format(disc_loss / (100 * 2)))
    else:
        discriminator_model.load_weights(discriminator_model_path)

    # TODO: what is an appropriate size for the image history buffer?
    image_history_buffer = ImageHistoryBuffer((0, img_height, img_width, img_channels), batch_size * 100, batch_size)

    combined_loss = np.zeros(shape=len(combined_model.metrics_names))
    disc_loss_real = np.zeros(shape=len(discriminator_model.metrics_names))
    disc_loss_refined = np.zeros(shape=len(discriminator_model.metrics_names))

    # see Algorithm 1 in https://arxiv.org/pdf/1612.07828v1.pdf
    for i in range(nb_steps):
        print('Step: {} of {}.'.format(i, nb_steps))

        # train the refiner
        for _ in range(k_g * 2):
            # sample a mini-batch of synthetic images
            synthetic_image_batch = get_image_batch(synthetic_generator)

            # update θ by taking an SGD step on mini-batch loss LR(θ)
            combined_loss = np.add(combined_model.train_on_batch(synthetic_image_batch,
                                                                 [synthetic_image_batch, y_real]), combined_loss)

        for _ in range(k_d):
            # sample a mini-batch of synthetic and real images
            synthetic_image_batch = get_image_batch(synthetic_generator)
            real_image_batch = get_image_batch(real_generator)

            # refine the synthetic images w/ the current refiner
            refined_image_batch = refiner_model.predict_on_batch(synthetic_image_batch)

            # use a history of refined images
            half_batch_from_image_history = image_history_buffer.get_from_image_history_buffer()
            image_history_buffer.add_to_image_history_buffer(refined_image_batch)

            if len(half_batch_from_image_history):
                refined_image_batch[:batch_size // 2] = half_batch_from_image_history

            # update φ by taking an SGD step on mini-batch loss LD(φ)
            disc_loss_real = np.add(discriminator_model.train_on_batch(real_image_batch, y_real), disc_loss_real)
            disc_loss_refined = np.add(discriminator_model.train_on_batch(refined_image_batch, y_refined),
                                       disc_loss_refined)

        if not i % log_interval:
            # plot batch of refined images w/ current refiner
            figure_name = 'refined_image_batch_step_{}.png'.format(i)
            print('Saving batch of refined images at adversarial step: {}.'.format(i))

            synthetic_image_batch = get_image_batch(synthetic_generator)
            plot_image_batch_w_labels.plot_batch(
                np.concatenate((synthetic_image_batch, refiner_model.predict_on_batch(synthetic_image_batch))),
                os.path.join(cache_dir, figure_name),
                label_batch=['Synthetic'] * batch_size + ['Refined'] * batch_size)

            # log loss summary
            print('Refiner model loss: {}.'.format(combined_loss / (log_interval * k_g * 2)))
            print('Discriminator model loss real: {}.'.format(disc_loss_real / (log_interval * k_d * 2)))
            print('Discriminator model loss refined: {}.'.format(disc_loss_refined / (log_interval * k_d * 2)))

            combined_loss = np.zeros(shape=len(combined_model.metrics_names))
            disc_loss_real = np.zeros(shape=len(discriminator_model.metrics_names))
            disc_loss_refined = np.zeros(shape=len(discriminator_model.metrics_names))

            # save model checkpoints
            model_checkpoint_base_name = os.path.join(cache_dir, '{}_model_step_{}.h5')
            refiner_model.save(model_checkpoint_base_name.format('refiner', i))
            discriminator_model.save(model_checkpoint_base_name.format('discriminator', i))


def main(synthesis_eyes_dir, mpii_gaze_dir, refiner_model_path, discriminator_model_path):
    adversarial_training(synthesis_eyes_dir, mpii_gaze_dir, refiner_model_path, discriminator_model_path)


if __name__ == '__main__':
    # TODO: if pre-trained models are passed in, we don't take the steps they've been trained for into account
    refiner_model_path = sys.argv[3] if len(sys.argv) >= 4 else None
    discriminator_model_path = sys.argv[4] if len(sys.argv) >= 5 else None

    main(sys.argv[1], sys.argv[2], refiner_model_path, discriminator_model_path)
sess = tf.Session()

# tf.train.SummaryWriter soon be deprecated, use following
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:  # tensorflow version < 0.12
    writer = tf.train.SummaryWriter('C://log', sess.graph)
else: # tensorflow version >= 0.12
    writer = tf.summary.FileWriter("C://log", sess.graph)

# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)