import os
import sys
import numpy as np
import h5py
import collections

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


Examples = collections.namedtuple("Examples", "names, pointsets_A, pointsets_B")
Examples1 = collections.namedtuple("Examples", "names, pointsets_A, pointsets_B")
Examples2 = collections.namedtuple("Examples", "names, pointsets_A, pointsets_B")
Examples3 = collections.namedtuple("Examples", "names, pointsets_A, pointsets_B")
Examples4 = collections.namedtuple("Examples", "names, pointsets_A")


def shuffle_examples( data ):

    idx = np.arange(  data.names.shape[0] )
    np.random.shuffle(idx)

    return Examples(
        names=data.names[idx, ...],
        pointsets_A=data.pointsets_A[idx, ...],
        pointsets_B=data.pointsets_B[idx, ...],
    )

def shuffle_examples_2files( data1, data2 ):

    idx = np.arange(  data1.names.shape[0] )
    np.random.shuffle(idx)

    return Examples1(
        names=data1.names[idx, ...],
        pointsets_A=data1.pointsets_A[idx, ...],
        pointsets_B=data1.pointsets_B[idx, ...],
    ), Examples2(
        names=data2.names[idx, ...],
        pointsets_A=data2.pointsets_A[idx, ...],
        pointsets_B=data2.pointsets_B[idx, ...],
    )

def shuffle_examples_3files( data1, data2, data3 ):

    idx = np.arange(  data1.names.shape[0] )
    np.random.shuffle(idx)

    return Examples1(
        names=data1.names[idx, ...],
        pointsets_A=data1.pointsets_A[idx, ...],
        pointsets_B=data1.pointsets_B[idx, ...],
    ), Examples2(
        names=data2.names[idx, ...],
        pointsets_A=data2.pointsets_A[idx, ...],
        pointsets_B=data2.pointsets_B[idx, ...],
    ), Examples3(
        names=data3.names[idx, ...],
        pointsets_A=data3.pointsets_A[idx, ...],
        pointsets_B=data3.pointsets_B[idx, ...],
    )

def shuffle_examples_3( data1, data2, data3 ):

    idx = np.arange(  data1.names.shape[0] )
    np.random.shuffle(idx)

    return Examples1(
        names=data1.names[idx, ...],
        pointsets_A=data1.pointsets_A[idx, ...],
        pointsets_B=data1.pointsets_B[idx, ...],
    ), Examples2(
        names=data2.names[idx, ...],
        pointsets_A=data2.pointsets_A[idx, ...],
        pointsets_B=data2.pointsets_B[idx, ...],
    ), Examples4(
        names=data3.names[idx, ...],
        pointsets_A=data3.pointsets_A[idx, ...],
    )


def load_examples(h5_filename,  fieldname_A, fieldname_B, fieldname_modelname ):
    print h5_filename
    f = h5py.File(h5_filename, 'r')
    pointsets_A = f[fieldname_A][:]
    pointsets_B = f[fieldname_B][:]
    names = f[fieldname_modelname][:]
    return Examples(
        names=names,
        pointsets_A=pointsets_A,
        pointsets_B=pointsets_B,
    )

def load_cormatrix(h5_filename,  fieldname_A, fieldname_modelname ):
    print h5_filename
    f = h5py.File(h5_filename, 'r')
    pointsets_A = f[fieldname_A][:]
    names = f[fieldname_modelname][:]
    return Examples4(
        names=names,
        pointsets_A=pointsets_A,
    )



def output_point_cloud_ply(xyzs, names, output_dir, foldername ):

    if not os.path.exists( output_dir ):
        os.mkdir(  output_dir  )

    plydir = output_dir + '/' + foldername

    if not os.path.exists( plydir ):
        os.mkdir( plydir )

    numFiles = len(names)

    for fid in range(numFiles):

        print('write: ' + plydir +'/'+names[fid]+'.ply')

        with open( plydir +'/'+names[fid]+'.ply', 'w') as f:
            pn = xyzs.shape[1]
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write('element vertex %d\n' % (pn) )
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            f.write('end_header\n')
            for i in range(pn):
                f.write('%f %f %f\n' % (xyzs[fid][i][0],  xyzs[fid][i][1],  xyzs[fid][i][2]) )

def output_weight_ply(xyzs, names, output_dir, foldername ):

    if not os.path.exists( output_dir ):
        os.mkdir(  output_dir  )

    plydir = output_dir + '/' + foldername

    if not os.path.exists( plydir ):
        os.mkdir( plydir )

    numFiles = len(names)

    for fid in range(numFiles):

        print('write: ' + plydir +'/'+names[fid]+'.ply')

        with open( plydir +'/'+names[fid]+'.ply', 'w') as f:
            pn = xyzs.shape[1]
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write('element vertex %d\n' % (pn) )
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            f.write('end_header\n')
            for i in range(pn):
                f.write('%f %f %f\n' % (xyzs[fid][i],  xyzs[fid][i],  xyzs[fid][i]) )                

def output_attentiion(att_array, names, output_dir, foldername ):

    if not os.path.exists( output_dir ):
        os.mkdir(  output_dir  )

    plydir = output_dir + '/' + foldername

    if not os.path.exists( plydir ):
        os.mkdir( plydir )

    numFiles = len(names)

    for fid in range(numFiles):

        print('write: ' + plydir +'/'+names[fid]+'.npy')

        np.save(plydir +'/'+names[fid]+'.npy', att_array[fid])

        #with open(plydir +'/'+names[fid]+'.npy', 'w') as f:
        #    f.write(att_array[fid][i])
