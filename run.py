import argparse
import subprocess
import tensorflow as tf
import numpy as np
from datetime import datetime
import json
import os
import sys
import datetime
import time
import collections

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# import P2PNET
# import P2PNET_inv
# import DCNET
# import MTNET
import MTNET_na
# import MT_NC_NET
# import MT_NC_NET
# import MTNET_na_cc
# import CA_MTNET
import ioUtil 

# DEFAULT SETTINGS
parser = argparse.ArgumentParser() 
# BASE_DIR = "/home/fangx2/data/code/unc_soft_tissue/unc_soft_tissue_prediction_program_and_data/hmri_data_hdf5"
# BASE_DIR = "/home/fangx2/data/code/unc_soft_tissue/unc_soft_tissue_prediction_program_and_data/Hdf5_File_220204"


BASE_DIR = "/home/fangx2/data/code/unc_soft_tissue/unc_soft_tissue_prediction_program_and_data/hdf5_data_40_4t"

parser.add_argument('--mode', type=str, default='train', help='train or test') 

parser.add_argument('--train_hdf5', default=(BASE_DIR +'/Face_change_trainX01_group03.hdf5' ))
parser.add_argument('--test_hdf5', default=(BASE_DIR +'/Face_change_testX01_group03.hdf5' ))
parser.add_argument('--trainldmk_hdf5', default=(BASE_DIR +'/Bone_vector_trainX01_group03.hdf5' ))
parser.add_argument('--testldmk_hdf5', default=(BASE_DIR +'/Bone_vector_testX01_group03.hdf5' ))


parser.add_argument('--gpu', type=int, default=1, help='which GPU to use [default: 0]')
parser.add_argument("--densityWeight", type=float, default=0.05, help="density weight [default: 1.0]")
parser.add_argument("--regularWeight", type=float, default=0, help="regularization weight [default: 0.1]")
parser.add_argument("--translossWeight", type=float, default=5, help="regularization weight [default: 0.1]")

parser.add_argument('--domain_A', default='pre', help='name of domain A')
parser.add_argument('--domain_B', default='post',  help='name of domain B')

parser.add_argument('--batch_size', type=int, default=2, help='Batch Size during training [default: 4]')
parser.add_argument('--epoch', type=int, default=500, help='number of epoches to run [default: 200]')
parser.add_argument('--decayEpoch',  type=int, default=50, help='steps(how many epoches) for decaying learning rate')
parser.add_argument("--nnk", type=int, default=8, help="density:  number of nearest neighbours [default: 8]")
parser.add_argument("--range_max", type=float, default=0.5, help="max length of point displacement[default: 1.0]")
parser.add_argument("--radiusScal", type=float, default=1.0, help="a constant for scaling radii in pointnet++ [default: 1.0]")
parser.add_argument("--noiseLength", type=int, default=32, help="length of point-wise noise vector [default: 32]")

#parser.add_argument('--checkpoint', default='/shenlab/lab_stor/leima/test_models/group05/epoch_120.ckpt', help='epoch_##.ckpt')
parser.add_argument('--checkpoint', default=None, help='epoch_##.ckpt')

###  None  None  None
parser.add_argument('--point_num', type=int, default=None, help='do not set the argument')
parser.add_argument('--ldmk_num', type=int, default=None, help='do not set the argument')
parser.add_argument('--example_num', type=int, default=None, help='do not set the argument')
parser.add_argument('--output_dir', type=str,  default='/home/fangx2/data/code/unc_soft_tissue/unc_soft_tissue_prediction_program_and_data/log_movement_transformer/log_0224_mtnc_40_4t/', help='do not set the argument')

parser.add_argument('--LEARNING_RATE', type=float, default=0.001, help='could be changed')
parser.add_argument('--ADAM_BETA_D', type=float, default=0.5, help='could be changed')
parser.add_argument('--ADAM_BETA_G', type=float, default=0.5, help='could be changed')  

FLAGS = parser.parse_args()

Train_examples = ioUtil.load_examples(FLAGS.train_hdf5, FLAGS.domain_A, FLAGS.domain_B, 'names')
Test_examples  = ioUtil.load_examples(FLAGS.test_hdf5, FLAGS.domain_A, FLAGS.domain_B, 'names')

Train_ldmk_examples = ioUtil.load_examples(FLAGS.trainldmk_hdf5, FLAGS.domain_A, FLAGS.domain_B, 'names')
Test_ldmk_examples  = ioUtil.load_examples(FLAGS.testldmk_hdf5, FLAGS.domain_A, FLAGS.domain_B, 'names')

# Train_facial_weights = ioUtil.load_examples(FLAGS.trainWeights_hdf5, FLAGS.domain_A, FLAGS.domain_B, 'names')
# Test_facial_weights = ioUtil.load_examples(FLAGS.testWeights_hdf5, FLAGS.domain_A, FLAGS.domain_B, 'names')


FLAGS.point_num = Train_examples.pointsets_A.shape[1]
POINT_NUM = FLAGS.point_num
FLAGS.ldmk_num = Train_ldmk_examples.pointsets_A.shape[1]
LDMK_NUM = FLAGS.ldmk_num


Example_NUM = Train_examples.pointsets_A.shape[0]
FLAGS.example_num = Example_NUM

TRAINING_EPOCHES = FLAGS.epoch

batch_size = FLAGS.batch_size 

if Train_examples.pointsets_B.shape[1] != POINT_NUM \
    or Test_examples.pointsets_A.shape[1] != POINT_NUM \
    or Test_examples.pointsets_B.shape[1] != POINT_NUM :
    print( 'point number inconsistent in the data set.')
    exit()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(FLAGS.gpu)  # specify which GPU(s) to be used

########## create output folders
current_time=datetime.datetime.now().strftime('%m%d%H%M')
datapath, basefname = os.path.split( FLAGS.train_hdf5 )
output_dir_name = 'FacialPrediction' + current_time + '_dw' + str(FLAGS.densityWeight)+ '_tsw' + str(FLAGS.translossWeight) + \
                    '_'+ FLAGS.train_hdf5[-21:-5]+ '_EndCat_EndAT_weights_AB'

output_dir = os.path.join(FLAGS.output_dir, output_dir_name)

MODEL_STORAGE_PATH = os.path.join(output_dir, 'trained_models')
if not os.path.exists(MODEL_STORAGE_PATH):
    os.makedirs(MODEL_STORAGE_PATH)

SUMMARIES_FOLDER = os.path.join(output_dir, 'summaries')
if not os.path.exists(SUMMARIES_FOLDER):
    os.mkdir(SUMMARIES_FOLDER)


########## Save test input
ioUtil.output_point_cloud_ply( Test_examples.pointsets_A,  Test_examples.names, output_dir, 'gt_face'+FLAGS.domain_A)
ioUtil.output_point_cloud_ply( Test_examples.pointsets_B, Test_examples.names, output_dir, 'gt_face'+FLAGS.domain_B)

ioUtil.output_point_cloud_ply( Test_ldmk_examples.pointsets_A,  Test_ldmk_examples.names, output_dir, 'gt_bone'+FLAGS.domain_A)
ioUtil.output_point_cloud_ply( Test_ldmk_examples.pointsets_B, Test_ldmk_examples.names, output_dir, 'gt_bone'+FLAGS.domain_B)

# ioUtil.output_point_cloud_ply( Test_facial_weights.pointsets_A,  Test_examples.names, output_dir, 'point_weights'+FLAGS.domain_A)
# ioUtil.output_point_cloud_ply( Test_facial_weights.pointsets_B, Test_examples.names, output_dir, 'point_weights'+FLAGS.domain_B)
# save training input
# ioUtil.output_point_cloud_ply( Train_examples.pointsets_A,  Train_examples.names, output_dir, 'gt_TrainPointA'+FLAGS.domain_A)
# ioUtil.output_point_cloud_ply( Train_examples.pointsets_B, Train_examples.names, output_dir, 'gt_TrainPointB'+FLAGS.domain_B)

# ioUtil.output_point_cloud_ply( Train_ldmk_examples.pointsets_A,  Train_ldmk_examples.names, output_dir, 'gt_TrainldmkA'+FLAGS.domain_A)
# ioUtil.output_point_cloud_ply( Train_ldmk_examples.pointsets_B, Train_ldmk_examples.names, output_dir, 'gt_TrainldmkB'+FLAGS.domain_B)

# print arguments
for k, v in FLAGS._get_kwargs():
    print(k + ' = ' + str(v) )


def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(FLAGS.gpu)):
            # model = P2PNET.create_model(FLAGS)
            # model = MT_NC_NET.create_model(FLAGS)
            # model = DCNET.create_model(FLAGS)
            # model = MTNET.create_model(FLAGS)
            model = MTNET_na.create_model(FLAGS)
            # model = MT_NC_NET.create_model(FLAGS)
            # model = MTNET_na_cc.create_model(FLAGS)
            # model = CA_MTNET.create_model(FLAGS)

        saver = tf.train.Saver( max_to_keep=100 )
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)

        init = tf.global_variables_initializer()
        sess.run(init)



        

        # Restore variables from disk.
        Start_epoch_number = 1
        if FLAGS.checkpoint is not None:
            print('load checkpoint: ' + FLAGS.checkpoint  )
            saver.restore(sess, FLAGS.checkpoint )

            fname = os.path.basename( FLAGS.checkpoint )
            Start_epoch_number = int( fname[6:-5] )  +  1

            print( 'Start_epoch_number = ' + str(Start_epoch_number) )



        fcmd = open(os.path.join(output_dir, 'arguments.txt'), 'w')
        fcmd.write(str(FLAGS))
        fcmd.close()
        
        with open(os.path.join(output_dir, 'loss_train.txt'), 'w') as file_train:
            file_train.write('total_data_loss_A' + ' ' + 'total_shape_loss_A' + ' '+ 'total_shape_error_A'+ ' ' + 'total_density_loss_A' +'\r\n')
                          
        with open(os.path.join(output_dir, 'loss_validation.txt'), 'w') as file_val:
            file_val.write('total_data_loss_A' + ' ' + 'total_shape_loss_A' + ' ' +  'total_shape_error_A' + ' ' + 'total_density_loss_A' + '\r\n')

        ########## Training one epoch  ##########

        def train_one_epoch(epoch_num):

            now = datetime.datetime.now()
            print(now.strftime("%Y-%m-%d %H:%M:%S"))
            start_time = time.time()

            for k, v in FLAGS._get_kwargs():
                print(k + ' = ' + str(v) )

            is_training = True

            #Train_examples_shuffled = Train_examples #ioUtil.shuffle_examples(Train_examples)
            #Train_ldmk_examples_shuffled = Train_ldmk_examples #ioUtil.shuffle_examples(Train_ldmk_examples)

            #Train_examples_shuffled, Train_ldmk_examples_shuffled = ioUtil.shuffle_examples_2files(Train_examples, Train_ldmk_examples)
            Train_examples_shuffled, Train_ldmk_examples_shuffled = ioUtil.shuffle_examples_2files(Train_examples, Train_ldmk_examples)

            pointsets_A = Train_examples_shuffled.pointsets_A
            pointsets_B = Train_examples_shuffled.pointsets_B
            ldmk_A=Train_ldmk_examples_shuffled.pointsets_A
            ldmk_B=Train_ldmk_examples_shuffled.pointsets_B
            # weight_A = Train_facial_weights_shuffled.pointsets_A
            # weight_B = Train_facial_weights_shuffled.pointsets_B

            names = Train_examples_shuffled.names

            num_data = pointsets_A.shape[0]
            num_batch = num_data // batch_size
            
            print('The value of num_batch:' + str( round(num_data//batch_size) ))
            print('The value of num_data:' + str( round(num_data) ))
            print('The value of batch_size:' + str( round(batch_size)))
            
            print(os.getcwd())


            total_data_loss_A = 0.0
            total_shape_loss_A = 0.0
            total_density_loss_A = 0.0
            total_Data_loss_All=0.0
            total_transform_loss_A=0.0
            total_shape_error_A=0.0
            # total_consistency_loss=0.0



            for j in range(num_batch):
                begidx = j * batch_size
                endidx = (j + 1) * batch_size

                feed_dict = {
                    model.pointSet_A_ph: pointsets_A[begidx: endidx, ...],
                    model.pointSet_B_ph: pointsets_B[begidx: endidx, ...],
                    model.ldmk_A_ph: ldmk_A[begidx: endidx, ...],
                    model.ldmk_B_ph: ldmk_B[begidx: endidx, ...],
                    # model.weights_A_ph: weight_A[begidx: endidx, ...],
                    # model.weights_B_ph: weight_B[begidx: endidx, ...],                    
                    model.is_training_ph: is_training,
                }

                fetches = {
                    "train": model.total_train,
                    "shapeLoss_A": model.shapeLoss_A,
                    "shape_error_A":model.shape_error_A,
                    "densityLoss_A": model.densityLoss_A,
                    "data_loss_A": model.data_loss_A,
                    "Dataloss_A": model.Dataloss_A,
                    "DataLoss": model.DataLoss,
                    "transform_loss_A":model.transform_loss_A,
                    # "consistency_loss":model.consistency_loss,
                    "learning_rate": model.learning_rate,
                    "global_step": model.global_step,
                }


                results = sess.run(fetches, feed_dict=feed_dict)

                total_data_loss_A += results["data_loss_A"]
                total_shape_loss_A += results["shapeLoss_A"]
                total_density_loss_A += results["densityLoss_A"]
                total_Data_loss_All +=results["Dataloss_A"]
                total_transform_loss_A +=results["transform_loss_A"]
                # total_consistency_loss +=results["consistency_loss"]
                total_shape_error_A +=results["shape_error_A"]
                




                if j % 50 == 0:
                    print('    ' + str(j) + '/' + str(num_batch) + ':    '  )
                    print('   Data_loss_All = {:.6f}'.format(results["Dataloss_A"] )   + \
                          '   data_loss_A = {:.6f},'.format(results["data_loss_A"] )  +  \
                          '   shape_loss = {:.6f},'.format(results["shapeLoss_A"] )   + \
                          '   shape_error = {:.6f},'.format(results["shape_error_A"] )   + \
                          '   density = {:.6f}'.format(results["densityLoss_A"] )   + \
                        #   '   consistency_loss = {:.6f}'.format(results["consistency_loss"] )   + \
                          '   transform_loss_A = {:.6f}'.format(results["transform_loss_A"] )
                          )


                    print('            learning_rate = {:.6f}'.format(results["learning_rate"] )  )
                    print('            global_step = {0}\n'.format(results["global_step"] )  )


            total_Data_loss_All    /=num_batch
            total_transform_loss_A /=num_batch
            total_data_loss_A      /= num_batch
            total_shape_loss_A  /= num_batch
            total_density_loss_A   /= num_batch
            total_shape_error_A /=num_batch
            # total_consistency_loss /=num_batch
            


            
            with open(os.path.join(output_dir, 'loss_train.txt'), 'a') as file_train:
                file_train.write(str(total_data_loss_A) + ' ' + str(total_shape_loss_A) + ' ' + str(total_shape_error_A) + ' ' + str(total_density_loss_A) +  '\r\n')

            print(  '\tEpoch Data_loss_All = %.8f,' % total_Data_loss_All    + \
                    '    train_data_loss_A = %.8f,' % total_data_loss_A    + \
                    '    transform_loss_A = %.8f,' % total_transform_loss_A    + \
                    '    shape = %.8f,' % total_shape_loss_A + \
                    # '    consistency = %.8f,' % total_consistency_loss + \
                    '    density = %.8f' % total_density_loss_A )

            elapsed_time = time.time() - start_time
            print( '\tply/sec:' + str( round(num_data/elapsed_time) ) )
            print( '\tduration of this epoch:' + str(round(elapsed_time/60) ) + ' min' )
            print( '\testimated finishing time:' + str(round(elapsed_time/60.0 * (TRAINING_EPOCHES-epoch_num-1)) ) + ' min' )
            


        ################## end  of train function #################### end  of train function ##########


        def eval_one_epoch(epoch_num, mustSavePly=False):
            is_training = False

            pointsets_A = Test_examples.pointsets_A
            pointsets_B = Test_examples.pointsets_B
            ldmk_A=Test_ldmk_examples.pointsets_A
            ldmk_B=Test_ldmk_examples.pointsets_B
            # weight_A = Test_facial_weights.pointsets_A
            # weight_B = Test_facial_weights.pointsets_B
            names = Test_examples.names

            num_data = pointsets_A.shape[0]
            num_batch = num_data // batch_size


            total_data_loss_A = 0.0
            total_shape_loss_A = 0.0
            total_density_loss_A = 0.0
            total_shape_error_A=0.0

            for j in range(num_batch):

                begidx = j * batch_size
                endidx = (j + 1) * batch_size

                feed_dict = {
                    model.pointSet_A_ph: pointsets_A[begidx: endidx, ...],
                    model.pointSet_B_ph: pointsets_B[begidx: endidx, ...],
                    model.ldmk_A_ph: ldmk_A[begidx: endidx, ...],
                    model.ldmk_B_ph: ldmk_B[begidx: endidx, ...],
                    # model.weights_A_ph: weight_A[begidx: endidx, ...],
                    # model.weights_B_ph: weight_B[begidx: endidx, ...],  
                    model.is_training_ph: is_training,
                }

                fetches = {
                    "shapeLoss_A": model.shapeLoss_A,
                    "shape_error_A":model.shape_error_A,
                    "densityLoss_A": model.densityLoss_A,
                    "data_loss_A": model.data_loss_A,
                    "Predicted_A": model.Predicted_A,
                    "displace_A2B": model.displace_A2B,
                    "beta": model.beta,
                    #"weight_output": model.weight_output,
                }


                results = sess.run(fetches, feed_dict=feed_dict)

                total_data_loss_A += results["data_loss_A"]
                total_shape_loss_A += results["shapeLoss_A"]
                total_density_loss_A += results["densityLoss_A"]
                total_shape_error_A +=results["shape_error_A"]



                # write test results
                if epoch_num  % 20 == 0  or  mustSavePly:
                    
                    # save predicted point sets with 1 single feeding pass
                    nametosave = names[begidx: endidx, ...]
                    Predicted_A_xyz = np.squeeze(np.array(results["Predicted_A"]))
                    Displacement_A_xyz=np.squeeze(np.array(results["displace_A2B"]))
                    beta_xyz = np.squeeze(np.array(results["beta"]))
                    #weights = np.squeeze(np.array(results["weight_output"]))
                    

                    ioUtil.output_point_cloud_ply(Predicted_A_xyz, nametosave, output_dir, 'Ep' + str(epoch_num) + FLAGS.domain_A  + '_predicted_Pointset')
                    ioUtil.output_point_cloud_ply(Displacement_A_xyz, nametosave, output_dir, 'Ep' + str(epoch_num) + FLAGS.domain_A  + '_predicted_Displacement')
                    #ioUtil.output_weight_ply(weights, nametosave, output_dir, 'Ep' + str(epoch_num) + FLAGS.domain_A  + '_predicted_weights')
                    #np.save(os.path.join(output_dir, str(num_batch) + 'attention.npy'), beta_xyz)
                    ioUtil.output_attentiion(beta_xyz, nametosave, output_dir, 'Ep' + str(epoch_num) + 'att_map')


                    # # save predicted point sets with 4 feeding passes
                    # for i in range(3):
                    #    results = sess.run(fetches, feed_dict=feed_dict)
                    #    Predicted_A_xyz__ = np.squeeze(np.array(results["Predicted_A"]))
                       
                    #    Predicted_A_xyz = np.concatenate((Predicted_A_xyz, Predicted_A_xyz__), axis=1)


                    #    #ioUtil.output_point_cloud_ply(Predicted_A_xyz, nametosave, output_dir,'Ep' + str(epoch_num) + '_predicted_' + FLAGS.domain_A + 'X4')
                    #    #ioUtil.output_point_cloud_ply(Predicted_B_xyz, nametosave, output_dir,'Ep' + str(epoch_num) + '_predicted_' + FLAGS.domain_B + 'X4')

                    # # save predicted point sets with 8 feeding passes
                    # for i in range(4):
                    #    results = sess.run(fetches, feed_dict=feed_dict)
                    #    Predicted_A_xyz__ = np.squeeze(np.array(results["Predicted_A"]))

                    #    Predicted_A_xyz = np.concatenate((Predicted_A_xyz, Predicted_A_xyz__), axis=1)


                    #    #ioUtil.output_point_cloud_ply( Predicted_A_xyz, nametosave, output_dir, 'Ep' + str(epoch_num) + '_predicted_' + FLAGS.domain_A + 'X8')
                    #    #ioUtil.output_point_cloud_ply( Predicted_B_xyz, nametosave, output_dir, 'Ep' + str(epoch_num) + '_predicted_' + FLAGS.domain_B + 'X8')



            total_data_loss_A      /= num_batch
            total_shape_loss_A  /= num_batch
            total_density_loss_A   /= num_batch
            total_shape_error_A /=num_batch

            print('\tEpoch_Eval_data_loss_A = %.6f,' % total_data_loss_A  + \
                  '    shape_loss = %.6f,' % total_shape_loss_A + \
                  '    shape_error = %.6f,' % total_shape_error_A + \
                  '    density_loss = %.6f' % total_density_loss_A)


            
            with open(os.path.join(output_dir, 'loss_validation.txt'), 'a') as file_val:
                file_val.write(str(total_data_loss_A) + ' ' + str(total_shape_loss_A) + ' ' + str(total_shape_error_A) + ' ' + str(total_density_loss_A) +  '\r\n')
                               
                               

        ################## end  of test function #################### end  of test function ##########

        if not os.path.exists(MODEL_STORAGE_PATH):
            os.mkdir(MODEL_STORAGE_PATH)

        if FLAGS.mode=='train':
            
            for epoch in range(Start_epoch_number,  TRAINING_EPOCHES+1):

                print( '\n>>> Training for the epoch %d/%d ...' % (epoch, TRAINING_EPOCHES))
                train_one_epoch(epoch)
                                
                if epoch % 20 == 0:

                    cp_filename = saver.save(sess, os.path.join(MODEL_STORAGE_PATH, 'epoch_' + str(epoch) + '.ckpt'))
                    print( 'Successfully store the checkpoint model into ' + cp_filename)

                    print('\n<<< Testing on the test dataset...')
                    eval_one_epoch(epoch, mustSavePly=True)
                else:
                    print('\n<<< Testing on the test dataset...')
                    eval_one_epoch(epoch, mustSavePly=False)

                    

        else:

            print( '\n<<< Testing on the test dataset ...')
            eval_one_epoch(Start_epoch_number, mustSavePly=True)



if __name__ == '__main__':
    train()
