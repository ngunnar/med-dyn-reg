import glob
import numpy as np
import pandas as pd
import scipy
from collections import defaultdict

from .data_utils import get_tf_multi_data

def read_image(image_data, rows, cols):
    """ read the image pixels (while converting bytearray from uint8 -> uint16)"""
    image_flat_u8 = image_data.TwoDSlicedata.Data
    image_flat_u16 = np.frombuffer( image_flat_u8.tobytes(), dtype=np.uint16)
    image = np.reshape(image_flat_u16, (rows, cols)) 
    return image

def read_direction_cosines(image_data):
    """ read the image directoin cosines """
    row = [ image_data.TwoDSlicedata.Orientation.RowDirectionCosines.X, image_data.TwoDSlicedata.Orientation.RowDirectionCosines.Y, image_data.TwoDSlicedata.Orientation.RowDirectionCosines.Z] 
    col = [ image_data.TwoDSlicedata.Orientation.ColumnDirectionCosines.X, image_data.TwoDSlicedata.Orientation.ColumnDirectionCosines.Y, image_data.TwoDSlicedata.Orientation.ColumnDirectionCosines.Z] 
    return np.concatenate((row, col))

def image_direction(image_data):
    """ determine which image plane """
    dir_cosines = read_direction_cosines(image_data)
    dir1, dir2, dir3 = [[1,  0,  0,  0,  0, -1], [ 0,  1,  0,  0,  0, -1], [1, 0, 0, 0, 1, 0]]
    if np.all(dir_cosines == dir1): return 0
    if np.all(dir_cosines == dir2): return 1
    if np.all(dir_cosines == dir3): return 2
    
    raise Exception('Illegal direction cosines {}'.format(dir_cosines))

class AckisLoader:
    @staticmethod
    def read_generator(patients, length):
        def gen():
            for p in patients:
                files = glob.glob(p + '**/cineData.mat')
                #print(p)
                for f in files:
                    #print(f)
                    mat = scipy.io.loadmat(f, squeeze_me=True, struct_as_record=False)
                    data = list(mat['cineData'])
                    data.sort(key = lambda x: x.TwoDSlicedata.Elapsed100NanosecondInterval)
                
                    rows = data[0].TwoDSlicedata.Dimension.Rows
                    cols = data[0].TwoDSlicedata.Dimension.Columns                
                    data_dict = defaultdict(list)
                    for i, d in enumerate(data):
                        data_dict['idx'].append(i)
                        data_dict['image'].append(read_image(d, rows, cols))
                        data_dict['orientation'].append(image_direction(d))
                        data_dict['time'].append(d.TwoDSlicedata.Elapsed100NanosecondInterval/(1e7))
                    df = pd.DataFrame(data_dict)
                    group_criterion = lambda x: x >= 0.3 # If time difference is larger than 300ms, we create a new sequence.
                    df['time_diff'] = df['time'].diff().fillna(0)                    
                    df['seq_group'] = (df['time_diff'].apply(group_criterion).cumsum())
                    groups = df.groupby('seq_group')
                    for _, g in groups:
                        transversals = []
                        sagittals = []
                        coronals = []                        
                        
                        for _, row in g.iterrows():
                            if row['orientation'] == 0:
                                coronals.append(row['image'])
                            if row['orientation'] == 1:
                                sagittals.append(row['image'])
                            if row['orientation'] == 2:
                                transversals.append(row['image'])

                        min_length = min(len(coronals), len(sagittals), len(transversals)) # 
                        if min_length < length:
                            continue
                        coronals = np.asarray(coronals[:min_length], dtype='float32')
                        sagittals = np.asarray(sagittals[:min_length], dtype='float32')
                        transversals = np.asarray(transversals[:min_length], dtype='float32')
                        #print(f, coronals.shape)
                        yield {'coronal': coronals, 'coronal_ref': coronals[0,...], 
                               'sagittal': sagittals, 'sagittal_ref': sagittals[0,...],
                               'transversal': transversals, 'transversal_ref': transversals[0,...]}
        return gen   

class AckisDataLoader:  
    def __init__(self, length, dim_y):
        patients = glob.glob('/data/Niklas/CinesMRL/*/')
        patients.sort()

        val_patients = ['1']
        test_patients = ['2', '3']

        self.train_patients = [p for p in patients if not (p.split('/')[-2] in val_patients or p.split('/')[-2] in test_patients)]
        self.val_patients = [p for p in patients if p.split('/')[-2] in val_patients]
        self.test_patients = [p for p in patients if p.split('/')[-2] in test_patients]

        #self.train_patients = patients[:int(len(patients)*0.9)] # First 90 %
        #self.val_patients = patients[int(len(patients)*0.9):int(len(patients)*0.95)] # Second 5 %
        #self.test_patients = patients[int(len(patients)*0.95):] # Last 5 %
                
        max_val = 3000
        min_val = 500
        
        self.train_data, self.train = get_tf_multi_data(self.train_patients, length, dim_y, max_val, lambda x : AckisLoader.read_generator(x, length), min_val)
        self.test_data, self.test = get_tf_multi_data(self.test_patients, length, dim_y, max_val, lambda x : AckisLoader.read_generator(x, length), min_val)
        self.val_data, self.val = get_tf_multi_data(self.val_patients, length, dim_y, max_val, lambda x : AckisLoader.read_generator(x, length), min_val)
        