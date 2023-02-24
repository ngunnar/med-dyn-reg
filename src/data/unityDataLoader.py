import numpy as np
import glob
import os
import SimpleITK as sitk
import ntpath
from tqdm import tqdm
import tensorflow as tf


cut2s = [os.path.join('V2_2035418238_ThoraxandPelvis','Fraction1'),
         os.path.join('V2_2035418238_ThoraxandPelvis','Fraction2'),
         os.path.join('V2_2035418238_ThoraxandPelvis','Fraction6'),
         os.path.join('V2_2035418238_ThoraxandPelvis','Fraction7'),
         os.path.join('V5_7446912650_PelvisThorax','Fraction1'),
         os.path.join('V5_7446912650_PelvisThorax','Fraction2'),
         os.path.join('V8_2197804674_PelvisAbdo','Fraction1'),
         os.path.join('V8_2197804674_PelvisAbdo','Fraction2')
        ]


test = {'comodo': ['24', '27'], 'volunteer': ['V7_2670993164_Abdo']}
evaluation = {'comodo': ['28'], 'volunteer': ['V8_2197804674']}

def get_filter(x):
    if 'Volunteer' in x:
        if any([c in x for c in cut2s]):
            return 'cut0001'
        else:
            return 'cut0000'
    else:
        return ''

    
def parse(idxs, i, n, base_path, path, sequence_glob, sequence_folder, frame_sort, split_filter, period, size):
    data_path = os.path.join(base_path, path)
    sequences = glob.glob(os.path.join(data_path, sequence_glob, ''))
    sequences = list(filter(split_filter, sequences))
    for sequence_path in sequences:
        sequence = glob.glob(os.path.join(sequence_path, sequence_folder, '*'))
        sequence = list(filter(lambda x: get_filter(x) in x, sequence))
        sequence.sort(key=frame_sort) 
        sequence = sequence[1:][::period]#first frame in each sequence is weird?
        grouped_sequence = [sequence[i:i + n] for i in range(0, len(sequence), n)]
        if len(grouped_sequence[-1]) < n:
            grouped_sequence.remove(grouped_sequence[-1])

        for g in grouped_sequence:
            idxs[i] = g
            i += 1
            if size is not None and len(idxs) >= size:
                return idxs, i
    return idxs, i


def get_kwargs(ds_type, split):
    if ds_type == 'comodo':
        if split == 'train':
            split_filter = lambda x: ntpath.basename(ntpath.dirname(x)) not in test['comodo'] and ntpath.basename(ntpath.dirname(x)) not in evaluation['comodo']
        elif split == 'test':
            split_filter = lambda x: ntpath.basename(ntpath.dirname(x)) in test['comodo']
        elif split == 'eval':
            split_filter = lambda x: ntpath.basename(ntpath.dirname(x)) in evaluation['comodo']
            
        path = os.path.join('comodo','comodo','comodo_pp')
        sequence_glob = '*'
        sequence_folder = 'bffe_cine_sagittal'
        frame_sort = lambda x: int(x[x.index('IM_')+3:].strip('.mha'))
        return path, sequence_glob, sequence_folder, frame_sort, split_filter 
    elif ds_type == 'volunteer':
        path = 'Volunteer'
        sequence_glob = os.path.join('*', 'Fraction*')
        sequence_folder = '2DSlices'
        if split == 'train':
            split_filter = lambda x: x.split(os.sep)[-3] not in test['volunteer'] and x.split(os.sep)[-3] not in evaluation['volunteer']
        elif split == 'test':
            split_filter = lambda x: x.split(os.sep)[-3] in test['volunteer']
        elif split == 'eval':
            split_filter = lambda x: x.split(os.sep)[-3] in evaluation['volunteer']

        frame_sort = lambda x: int(ntpath.basename(x).split('_')[0])
        return path, sequence_glob, sequence_folder, frame_sort, split_filter

class TensorflowDatasetLoader():
    def __init__(self, datasets=['comodo','volunteer'], 
                 root = '/data/Niklas/Unity', 
                 split='train', 
                 image_shape = (112,112),
                 length = 50,
                 period = 1,
                 size = None):
        self.idxs = {}
        n = length
        self.period = period
        self.image_shape = image_shape
        i = 0
        for ds in datasets:
            path, sequence_glob, sequence_folder, frame_sort, split_filter = get_kwargs(ds, split)
            self.idxs, i = parse(self.idxs, i, n, root, path, sequence_glob, sequence_folder, frame_sort, split_filter, self.period, size)
            if size is not None and len(self.idxs) >= size:
                break
            
        data = tf.data.Dataset.from_generator(
            self.generator(),
            output_types = ({"input_video": tf.float32, "input_mask": tf.bool}),
            output_shapes = ({"input_video": (length, *self.image_shape), "input_mask": (length)}))        
        self.data = data
        self.resizing = tf.keras.layers.experimental.preprocessing.Resizing(self.image_shape[0], self.image_shape[1], interpolation='bilinear')
    
    def _read_video(self, idx):
        files = self.idxs[idx]
        video = []        
        for f in files:
            image = sitk.ReadImage(f)
            image = sitk.GetArrayFromImage(image)
            if len(image.shape) == 3:
                image = image[0,...]
            video.append(image)
        video = np.asarray(video)
        video = self.resizing(video[...,None])
        video = tf.clip_by_value(video, clip_value_min=0, clip_value_max=4000)
        video = tf.keras.layers.experimental.preprocessing.Rescaling(1. / tf.reduce_max(video), offset=0.0)(video)
        video = video[...,0]
        mask = np.zeros(video.shape[0], dtype='bool')
        return video, mask, video[0,...]
    
    def generator(self):
        def gen():
            for idx in self.idxs:
                video, mask, _ = self._read_video(idx)
                if np.any(mask == True):
                    tqdm.write(idx)                           
                yield {'input_video': video, 'input_mask': mask}
        return gen