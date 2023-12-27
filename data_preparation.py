import os
import numpy as np
import h5py
import laspy
from data_utils.indoor3d_util import room2blocks_wrapper_normalized

# Constants for data preparation
args_num_class = 2             # number of class
NUM_POINT = 4096               # number of point for each block
block_size = 25                # size of  sliding window in points
stride = 5                     # step size, overlap between blocks
sample_num = 10000             # maximum number of blocks
random_sample = False
min_point_discard_block = 20   # minimum number of points required for a block to be considered valid

H5_BATCH_SIZE = 1000
data_dim = [NUM_POINT, 9] # XYZ, RGB, normXYZ
label_dim = [NUM_POINT]
data_dtype = 'float32'
label_dtype = 'uint8'

output_h5 = 'data/sem_seg_hdf5_data'
data_folder = 'data/data'
output_folder = 'data/sem_seg_data'
list_path = 'data/list.txt'
filelist_npy = 'data/npy_data_list.txt'
data_dir = 'data/sem_seg_data'
DATA_DIR = 'data'

# Read txt file, shift coordinates and save as numpy
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

filelist = [line.rstrip() for line in open(list_path)]
filepath_list = [os.path.join(data_folder, p).replace("\\", "/") for p in filelist]
print(filepath_list)
for filepath in filepath_list:
    elements = filepath.split('/')
    out_filename = os.path.join(output_folder,(elements[-1]).split('.')[0]+'.npy')
    print(out_filename)

    lasfile = laspy.read(filepath)
    x = lasfile.x
    y = lasfile.y
    z = lasfile.z
    r = lasfile.red
    g = lasfile.green
    b = lasfile.blue
    C = lasfile.classification

    data_label = np.column_stack((x, y, z, r, g, b, C))

    xy_min = np.amin(data_label, axis=0)[0:2]
    print(xy_min)

    data_label[:, 0:2] -= xy_min
    print(data_label[1,0:2])

    np.save(out_filename, data_label)

# Write numpy array data and label to h5_filename
def save_h5(h5_filename, data, label, data_dtype='uint8', label_dtype='uint8'):
    h5_fout = h5py.File(h5_filename, 'w') # add 'w' to write
    h5_fout.create_dataset(
            'data', data=data,
            compression='gzip', compression_opts=4,
            dtype=data_dtype)
    h5_fout.create_dataset(
            'label', data=label,
            compression='gzip', compression_opts=1,
            dtype=label_dtype)
    h5_fout.close()

# Set paths
data_label_files = [os.path.join(data_dir, line.rstrip()) for line in open(filelist_npy)]
output_dir = output_h5
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
output_filename_prefix = os.path.join(output_dir, 'ply_data_all')
output_room_filelist = os.path.join(output_dir, 'room_filelist.txt')
output_all_file = os.path.join(output_dir, 'all_files.txt')
fout_room = open(output_room_filelist, 'w')
all_file = open(output_all_file, 'w')

# Create blocks
# Adopted from: https://github.com/charlesq34/pointnet
# Generate blocks for training data
# Blocks are stored in h5
# --------------------------------------
# ----- BATCH WRITE TO HDF5 -----
# --------------------------------------
batch_data_dim = [H5_BATCH_SIZE] + data_dim
batch_label_dim = [H5_BATCH_SIZE] + label_dim
h5_batch_data = np.zeros(batch_data_dim, dtype = np.float32)
h5_batch_label = np.zeros(batch_label_dim, dtype = np.uint8)
buffer_size = 0  # state: record how many samples are currently in buffer
h5_index = 0 # state: the next h5 file to save

def check_block(data_label, block_size, stride):
    data_label = np.load(data_label_filename)
    data = data_label[:, 0:-1]
    limit = np.amax(data, 0)[0:3]
    num_block_x = int(np.ceil((limit[0] - block_size) / stride)) + 1
    num_block_y = int(np.ceil((limit[1] - block_size) / stride)) + 1
    print('num_block_x', num_block_x)
    print('num_block_y', num_block_y)

    if num_block_x <= 0 or num_block_y <= 0:
        return False  # Area cannot be split into blocks

    return True

def insert_batch(data, label, last_batch=False):
    global h5_batch_data, h5_batch_label
    global buffer_size, h5_index
    data_size = data.shape[0]
    # If there is enough space, just insert
    if buffer_size + data_size <= h5_batch_data.shape[0]:
        h5_batch_data[buffer_size:buffer_size+data_size, ...] = data
        h5_batch_label[buffer_size:buffer_size+data_size] = label
        buffer_size += data_size
    else: # not enough space
        capacity = h5_batch_data.shape[0] - buffer_size
        assert(capacity>=0)
        if capacity > 0:
           h5_batch_data[buffer_size:buffer_size+capacity, ...] = data[0:capacity, ...]
           h5_batch_label[buffer_size:buffer_size+capacity, ...] = label[0:capacity, ...]
        # Save batch data and label to h5 file, reset buffer_size
        h5_filename =  output_filename_prefix + '_' + str(h5_index) + '.h5'
        save_h5(h5_filename, h5_batch_data, h5_batch_label, data_dtype, label_dtype)
        print('Stored {0} with size {1}'.format(h5_filename, h5_batch_data.shape[0]))
        h5_index += 1
        buffer_size = 0
        # recursive call
        insert_batch(data[capacity:, ...], label[capacity:, ...], last_batch)
    if last_batch and buffer_size > 0:
        h5_filename =  output_filename_prefix + '_' + str(h5_index) + '.h5'
        save_h5(h5_filename, h5_batch_data[0:buffer_size, ...], h5_batch_label[0:buffer_size, ...], data_dtype, label_dtype)
        print('Stored {0} with size {1}'.format(h5_filename, buffer_size))
        h5_index += 1
        buffer_size = 0
    return

blockability_dict = {}
for i, data_label_filename in enumerate(data_label_files):
    print(data_label_filename)
    is_blockable = check_block(data_label_filename, block_size=block_size, stride=stride)
    blockability_dict[data_label_filename] = is_blockable

for area, is_blockable in blockability_dict.items():
    if is_blockable:
        print(f"{area} can be split into blocks.")
    else:
        print(f"{area} cannot be split into blocks.")

sample_cnt = 0
for i, data_label_filename in enumerate(data_label_files):
    print(data_label_filename)
    data, label = room2blocks_wrapper_normalized(data_label_filename, NUM_POINT, block_size=block_size, stride=stride,
                                                         random_sample=random_sample, sample_num=sample_num)
    print('{0}, {1}'.format(data.shape, label.shape))
    for _ in range(data.shape[0]):
        fout_room.write(os.path.basename(data_label_filename)[0:-4]+'\n')

    sample_cnt += data.shape[0]
    insert_batch(data, label, i == len(data_label_files)-1)

fout_room.close()
print("Total samples: {0}".format(sample_cnt))

for i in range(h5_index):
    all_file.write(os.path.join(output_h5[5:], 'ply_data_all_') + str(i) +'.h5\n') # Check output name
all_file.close()

# Check h5 files
h5_filename = 'data/sem_seg_hdf5_data/ply_data_all_0.h5'
check_h5 = h5py.File(h5_filename, 'r')
print(check_h5.keys())
print(check_h5['data'])
print(check_h5['label'])
print('The data = ', check_h5['data'][0,100,:])
print('The label = ', check_h5['label'][0,100])
check_h5.close()