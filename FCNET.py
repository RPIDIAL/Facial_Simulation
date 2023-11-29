import os
import sys
import collections
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(   BASE_DIR + "/utils")
sys.path.append(   BASE_DIR + "/tf_ops")
sys.path.append(   BASE_DIR + "/tf_ops/3d_interpolation")
sys.path.append(   BASE_DIR + "/tf_ops/grouping")
sys.path.append(   BASE_DIR + "/tf_ops/sampling")
import tensorflow as tf
import numpy as np
import open3d as o3d
import tf_util
from pointnet_util import pointnet_sa_module, pointnet_fp_module, pointnet_sa_module_msg_attention

from PointConv import feature_encoding_layer, feature_decoding_layer_depthwise
from self_attention import attention_unit

#weight_decay = None
BANDWIDTH = 0.05 #BANDWIDTH


Model = collections.namedtuple("Model", \
                               "pointSet_A_ph,  pointSet_B_ph, \
                               ldmk_A_ph,  ldmk_B_ph,\
                               is_training_ph,\
                               Predicted_A, displace_A2B, beta, shape_error_A,\
                               data_loss_A, shapeLoss_A, densityLoss_A, \
                               Dataloss_A, DataLoss, transform_loss_A, \
                               total_train,\
                               learning_rate,  global_step,  bn_decay"                    )

def create_model(FLAGS):

    ############################################################
    ####################  Hyper-parameters   ####################
    ##############################################################

    global_step = tf.train.get_or_create_global_step()

    learning_rate = tf.train.exponential_decay(
        FLAGS.LEARNING_RATE,  # base learning rate
        global_step   * FLAGS.batch_size,  # global_var indicating the number of steps
        FLAGS.example_num  * FLAGS.decayEpoch,  # step size
        0.5,  # decay rate
        staircase=True
    )
    learning_rate = tf.maximum(learning_rate, 1e-4)

    bn_momentum = tf.train.exponential_decay(
        0.5,
        global_step  * FLAGS.batch_size,  # global_var indicating the number of steps
        FLAGS.example_num * FLAGS.decayEpoch * 2,     # step size,
        0.5,   # decay rate
        staircase=True
    )
    bn_decay = tf.minimum(0.99,   1 - bn_momentum)


    ##############################################################
    ####################  Create the network  ####################
    ##############################################################

    pointSet_A_ph = tf.placeholder( tf.float32, shape=(FLAGS.batch_size, FLAGS.point_num, 3) )
    pointSet_B_ph = tf.placeholder( tf.float32, shape=(FLAGS.batch_size, FLAGS.point_num, 3) )
    ldmk_A_ph = tf.placeholder( tf.float32, shape=(FLAGS.batch_size, FLAGS.ldmk_num, 3) )
    ldmk_B_ph = tf.placeholder( tf.float32, shape=(FLAGS.batch_size, FLAGS.ldmk_num, 3) )
    # weights_A_ph = tf.placeholder( tf.float32, shape=(FLAGS.batch_size, FLAGS.point_num, 3) )
    # weights_B_ph = tf.placeholder( tf.float32, shape=(FLAGS.batch_size, FLAGS.point_num, 3) )

    is_training_ph = tf.placeholder( tf.bool, shape=() )

    #ldmk_features = encoder(ldmk_B_ph, FLAGS, is_training_ph, bn_decay)
    #ldmk_features = encoder(ldmk_A_ph, ldmk_B_ph, FLAGS, is_training_ph, bn_decay)
    Bone_features=BoneVector_encoder(ldmk_A_ph, ldmk_B_ph, FLAGS, is_training_ph, bn_decay)
    # Bone_features=BoneVector_encoder(pointSet_B_ph, pointSet_A_ph, FLAGS, is_training_ph, bn_decay)

    noise1 = None
    if FLAGS.noiseLength > 0:
        noise1 = tf.random_normal(shape=[FLAGS.batch_size, FLAGS.point_num, FLAGS.noiseLength], mean=0.0, stddev=1.0, dtype=tf.float32)

    with tf.variable_scope("p2pnet_A2B") as scope:
        displace_A2B, beta = get_displacements(pointSet_A_ph, Bone_features, is_training_ph, noise1, BANDWIDTH, FLAGS, bn_decay=None)

    Predicted_A = pointSet_A_ph + displace_A2B

    data_loss_A, shapeLoss_A, densityLoss_A, shape_error_A = get_Geometric_Loss(Predicted_A, pointSet_B_ph, FLAGS)

    # shapeLoss_A=chamfer_distance_tf(Predicted_A, pointSet_B_ph, weights_A_ph, weights_B_ph)
    #data_loss_A=shapeLoss_A + densityLoss_A * FLAGS.densityWeight

    transform_loss_A = get_transform_Loss(Predicted_A, pointSet_A_ph, FLAGS)
    Dataloss_A=shapeLoss_A + densityLoss_A * FLAGS.densityWeight + transform_loss_A*FLAGS.translossWeight
    
    DataLoss = Dataloss_A
    train_variables = tf.trainable_variables()

    #trainer = tf.train.AdamOptimizer(learning_rate)
    #total_train_op = trainer.minimize(DataLoss, var_list=train_variables, global_step=global_step)

    total_train_op = tf.train.AdamOptimizer(learning_rate, beta1=FLAGS.ADAM_BETA_G, beta2=0.9).minimize(DataLoss, var_list=train_variables, global_step=global_step)
    total_train = total_train_op

    ##############################################################
    ####################  Create summarizers  ####################
    ##############################################################

    # train_dataloss_A_ph = tf.placeholder(tf.float32, shape=())
    # train_dataloss_B_ph = tf.placeholder(tf.float32, shape=())
    # train_regul_ph = tf.placeholder(tf.float32, shape=())

    # test_dataloss_A_ph = tf.placeholder(tf.float32, shape=())
    # test_dataloss_B_ph = tf.placeholder(tf.float32, shape=())
    # test_regul_ph = tf.placeholder(tf.float32, shape=())


    # lr_sum_op = tf.summary.scalar('learning rate', learning_rate)
    # global_step_sum_op = tf.summary.scalar('batch_number', global_step)

    # train_dataloss_A_sum_op = tf.summary.scalar('train_dataloss_A', train_dataloss_A_ph)
    # train_dataloss_B_sum_op = tf.summary.scalar('train_dataloss_B', train_dataloss_B_ph)
    # train_regul_sum_op = tf.summary.scalar('train_regul', train_regul_ph)

    # test_dataloss_A_sum_op = tf.summary.scalar('test_dataloss_A', test_dataloss_A_ph)
    # test_dataloss_B_sum_op = tf.summary.scalar('test_dataloss_B', test_dataloss_B_ph)
    # test_regul_sum_op = tf.summary.scalar('test_regul', test_regul_ph)


    # training_sum_ops = tf.summary.merge( \
    #     [lr_sum_op, train_dataloss_A_sum_op, train_dataloss_B_sum_op, train_regul_sum_op])

    # testing_sum_ops = tf.summary.merge( \
    #     [test_dataloss_A_sum_op, test_dataloss_B_sum_op, test_regul_sum_op ]) 

    return Model(
        pointSet_A_ph=pointSet_A_ph,  pointSet_B_ph=pointSet_B_ph,
        ldmk_A_ph=ldmk_A_ph,  ldmk_B_ph=ldmk_B_ph,
        # weights_A_ph=weights_A_ph, weights_B_ph=weights_B_ph,
        is_training_ph=is_training_ph,
        Predicted_A=Predicted_A, beta=beta, displace_A2B=displace_A2B, shape_error_A=shape_error_A,
        Dataloss_A=Dataloss_A, DataLoss = DataLoss, transform_loss_A=transform_loss_A,
        data_loss_A=data_loss_A,   shapeLoss_A=shapeLoss_A,     densityLoss_A=densityLoss_A,
        total_train=total_train,     
        learning_rate=learning_rate, global_step=global_step, bn_decay=bn_decay
    )



def BoneVector_encoder(input_pointsA, input_pointsB, FLAGS, is_training, bn_decay=None):

    batch_size = FLAGS.batch_size
    num_points = FLAGS.point_num

    l0_xyz = input_pointsA
    l0_points = input_pointsB #Vector


    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=1024, radius=0.1 * FLAGS.radiusScal, nsample=64,
                                                       mlp=[64, 64, 128], mlp2=None, group_all=False,
                                                       is_training=is_training, bn_decay=bn_decay, scope='layer11')
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=384, radius=0.2* FLAGS.radiusScal, nsample=64,
                                                       mlp=[128, 128, 128], mlp2=None, group_all=False,
                                                       is_training=is_training, bn_decay=bn_decay, scope='layer12')
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None,
                                                       mlp=[128, 128, 128], mlp2=None, group_all=True,
                                                       is_training=is_training, bn_decay=bn_decay, scope='layer13')
    
    ##### Attention for bony encoder
    # l3_points = tf.expand_dims(l3_points,axis=2)
    # l3_points = attention_unit(l3_points, is_training=is_training) 
    
    l3_points = tf.squeeze(l3_points)
    return l3_points

def get_displacements(input_points, ldmk_features, is_training, noise, sigma, FLAGS, bn_decay=None, weight_decay=None):
    """ Semantic segmentation PointNet, input is BxNx3, output Bxnum_class """

    batch_size = FLAGS.batch_size
    num_points = FLAGS.point_num

    point_cloud = input_points

    l0_xyz = point_cloud
    l0_points = point_cloud



    # # Feature encoding layers
    # l1_xyz, l1_points = feature_encoding_layer(l0_xyz, l0_points, npoint=1024, radius = 0.1, sigma = sigma, K=32, mlp=[64,64,128], is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer1')
    # l2_xyz, l2_points = feature_encoding_layer(l1_xyz, l1_points, npoint=256, radius = 0.2, sigma = 2 * sigma, K=32, mlp=[128,128,256], is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer2')
    # l3_xyz, l3_points = feature_encoding_layer(l2_xyz, l2_points, npoint=64, radius = 0.4, sigma = 4 * sigma, K=32, mlp=[256,256,512], is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer3')
    # l4_xyz, l4_points = feature_encoding_layer(l3_xyz, l3_points, npoint=36, radius = 0.8, sigma = 8 * sigma, K=32, mlp=[512,512,1024], is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer4')

    # # Feature decoding layers
    # l3_points = feature_decoding_layer_depthwise(l3_xyz, l4_xyz, l3_points, l4_points, 0.8, 8 * sigma, 16, [512,512], is_training, bn_decay, weight_decay, scope='fa_layer1')
    # l2_points = feature_decoding_layer_depthwise(l2_xyz, l3_xyz, l2_points, l3_points, 0.4, 4 * sigma, 16, [256,256], is_training, bn_decay, weight_decay, scope='fa_layer2')
    # l1_points = feature_decoding_layer_depthwise(l1_xyz, l2_xyz, l1_points, l2_points, 0.2, 2 * sigma, 16, [256,128], is_training, bn_decay, weight_decay, scope='fa_layer3')
    # l0_points = feature_decoding_layer_depthwise(l0_xyz, l1_xyz, l0_points, l1_points, 0.1, sigma, 16, [128,128,128], is_training, bn_decay, weight_decay, scope='fa_layer4')

    
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=1024, radius=0.1 * FLAGS.radiusScal, nsample=64,
                                                       mlp=[64, 64, 128], mlp2=None, group_all=False,
                                                       is_training=is_training, bn_decay=bn_decay, scope='layer1')
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=384, radius=0.2* FLAGS.radiusScal, nsample=64,
                                                       mlp=[128, 128, 256], mlp2=None, group_all=False,
                                                       is_training=is_training, bn_decay=bn_decay, scope='layer2')
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=128, radius=0.4* FLAGS.radiusScal, nsample=64,
                                                       mlp=[256, 256, 512], mlp2=None, group_all=False,
                                                       is_training=is_training, bn_decay=bn_decay, scope='layer3')

    # PointNet
    l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=None, radius=None, nsample=None,
                                                       mlp=[512, 512, 1024], mlp2=None, group_all=True,
                                                       is_training=is_training, bn_decay=bn_decay, scope='layer4')

    # ldmk_features_mid = tf.tile(tf.expand_dims(ldmk_features, 1), [1, 1, 1])
    # l4_points = tf.concat([l4_points, ldmk_features_mid], axis=-1)


    #l4_xyz, l4_points, l4_indices = pointnet_sa_module(l4_xyz, l4_points, npoint=None, radius=None, nsample=None,
    #                                                   mlp=[512, 512, 512], mlp2=None, group_all=True,
    #                                                   is_training=is_training, bn_decay=bn_decay, scope='layer5')


    ### Feature Propagation layers  #################  featrue maps are interpolated according to coordinate  ################     
    l3_points = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [512, 512], is_training, bn_decay, scope='fa_layer1')
    l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [512, 256], is_training, bn_decay, scope='fa_layer2')
    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256, 128], is_training, bn_decay, scope='fa_layer3')
    l0_points = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points, [128, 128, 128], is_training, bn_decay, scope='fa_layer4')

    ## Attention model
    l0_points = tf.expand_dims(l0_points,axis=2)
    l0_points, beta = attention_unit(l0_points, is_training=is_training)
    l0_points = tf.squeeze(l0_points)

    ldmk_features_end = tf.tile(tf.expand_dims(ldmk_features, 1), [1,num_points, 1])
    l0_points = tf.concat([l0_points, ldmk_features_end], axis=-1)

    # if noise is not None:
    #    l0_points = tf.concat(axis=2, values=[l0_points, noise])

    net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay )
    net = tf_util.conv1d(net, 64, 1, padding='VALID', bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    net = tf_util.conv1d(net, 3, 1, padding='VALID', activation_fn=None, scope='fc3')

    displacements = tf.sigmoid(net) * FLAGS.range_max * 2 - FLAGS.range_max

    return displacements, beta

#def get_surface_loss(predictedPts, targetpoints, FLAGS):


def pairwise_l2_norm2_batch(x, y, scope=None):
    with tf.op_scope([x, y], scope, 'pairwise_l2_norm2_batch'):
        nump_x = tf.shape(x)[1]
        nump_y = tf.shape(y)[1]

        xx = tf.expand_dims(x, -1)
        xx = tf.tile(xx, tf.stack([1, 1, 1, nump_y]))

        yy = tf.expand_dims(y, -1)
        yy = tf.tile(yy, tf.stack([1, 1, 1, nump_x]))
        yy = tf.transpose(yy, perm=[0, 3, 2, 1])

        diff = tf.subtract(xx, yy)
        square_diff = tf.square(diff)

        square_dist = tf.reduce_sum(square_diff, 2)

        return square_dist


def get_Geometric_Loss(predictedPts, targetpoints, FLAGS):

    # calculate shape loss
    square_dist = pairwise_l2_norm2_batch(targetpoints, predictedPts)
    dist = tf.sqrt( square_dist )
    minRow = tf.reduce_min(dist, axis=2)
    minCol = tf.reduce_min(dist, axis=1)
    # weight_A=tf.reduce_mean(weights_A_ph,2)
    # weight_B=tf.reduce_mean(weights_B_ph,2)
    # minRow_weight=tf.multiply(minRow,weight_B)
    # minCol_weight=tf.multiply(minCol,weight_A)

    shape_error = (tf.reduce_mean(minRow) + tf.reduce_mean(minCol))*125
    #shapeLoss = tf.reduce_mean(minRow_weight) + tf.reduce_mean(minCol_weight)
    shapeLoss = (tf.reduce_mean(minRow) + tf.reduce_mean(minCol))

    #print('shape loss:', shapeLoss)

    # calculate density loss
    square_dist2 = pairwise_l2_norm2_batch(targetpoints, targetpoints)
    dist2 = tf.sqrt(square_dist2)
    knndis = tf.nn.top_k(tf.negative(dist), k=FLAGS.nnk)
    knndis2 = tf.nn.top_k(tf.negative(dist2), k=FLAGS.nnk)
    densityLoss = tf.reduce_mean(tf.abs(knndis.values - knndis2.values))

	
    data_loss = shapeLoss + densityLoss * FLAGS.densityWeight
    return data_loss, shapeLoss, densityLoss, shape_error

def get_transform_Loss(outputpoints, inputpoints, FLAGS):

    # # calculate density loss
    # square_dist2 = pairwise_l2_norm2_batch(targetpoints, targetpoints)
    # dist2 = tf.sqrt(square_dist2)
    # knndis = tf.nn.top_k(tf.negative(dist), k=FLAGS.nnk)
    # knndis2 = tf.nn.top_k(tf.negative(dist2), k=FLAGS.nnk)
    # densityLoss = tf.reduce_mean(tf.abs(knndis.values - knndis2.values))

    # data_loss = shapeLoss + densityLoss * FLAGS.densityWeight
    # return data_loss, shapeLoss, densityLoss, shape_error

    bs = FLAGS.batch_size
    npts = FLAGS.point_num
    R = FLAGS.radiusScal *0.02

    square_dist1 = pairwise_l2_norm2_batch(inputpoints, inputpoints)
    dist1 = tf.sqrt(square_dist1+1e-12) #[batch_size, npts, npts]
    #dist1 = square_dist1
    mask = dist1 < R

    square_dist2 = pairwise_l2_norm2_batch(outputpoints, outputpoints)
    dist2 = tf.sqrt(square_dist2+1e-12)
    #dist2 = square_dist2

    sub_dist = tf.abs(dist2-dist1)
    dist_nei = tf.boolean_mask(sub_dist, mask)

    mean_dist = tf.reduce_mean(dist_nei)

    return mean_dist

def get_Regularizing_Loss(pointSet_A_ph, pointSet_B_ph,  Predicted_A, Predicted_B):

    displacements_A = tf.concat(axis=2, values=[pointSet_A_ph, Predicted_B])
    displacements_B = tf.concat(axis=2, values=[Predicted_A,   pointSet_B_ph])

    square_dist = pairwise_l2_norm2_batch( displacements_A,   displacements_B )
    dist = tf.sqrt(square_dist)

    minRow = tf.reduce_min(dist, axis=2)
    minCol = tf.reduce_min(dist, axis=1)
    RegularLoss = (tf.reduce_mean(minRow) + tf.reduce_mean(minCol))/2

    return RegularLoss


def distance_matrix(array1, array2):
    """
    arguments: 
        array1: the array, size: (num_point, num_feature)
        array2: the samples, size: (num_point, num_feature)
    returns:
        distances: each entry is the distance from a sample to array1
            , it's size: (num_point, num_point)
    """
    num_point, num_features = array1.shape
    expanded_array1 = tf.tile(array1, (num_point, 1))
    expanded_array2 = tf.reshape(
            tf.tile(tf.expand_dims(array2, 1), 
                    (1, num_point, 1)),
            (-1, num_features))
    distances = tf.norm(expanded_array1-expanded_array2, axis=1)
    distances = tf.reshape(distances, (num_point, num_point))
    return distances

# def av_dist(array1, array2, weights):
#     """
#     arguments:
#         array1, array2: both size: (num_points, num_feature)
#     returns:
#         distances: size: (1,)
#     """
#     distances = distance_matrix(array1, array2)
#     distances = tf.reduce_min(distances, axis=1)
#     weights = tf.reduce_mean(weights,1)
#     distances = tf.multiply(distances,weights)
#     distances = tf.reduce_mean(distances)
#     return distances

# def av_dist_sum(arrays):
#     """
#     arguments:
#         arrays: array1, array2
#     returns:
#         sum of av_dist(array1, array2) and av_dist(array2, array1)
#     """
#     array1, array2, weights_A_ph, weights_B_ph= arrays
#     av_dist1 = av_dist(array1, array2, weights_A_ph)
#     av_dist2 = av_dist(array2, array1, weights_B_ph)
#     return av_dist1+av_dist2

# def chamfer_distance_tf(array1, array2, weights_A_ph, weights_B_ph):
#     batch_size, num_point, num_features = array1.shape
#     dist = tf.reduce_mean(
#                tf.map_fn(av_dist_sum, elems=(array1, array2, weights_A_ph, weights_B_ph), dtype=tf.float32)
#            )
#     return dist
