import numpy as np
import os
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
from create_imagenet import convert_path_to_npy
import cv2
from bucket import bucket
import torch
import torch.utils.data
import torchvision

def _process_path(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)
    img = np.transpose(img, (2, 0, 1))
    tensor = torch.tensor(img / 255)
    return tensor.unsqueeze(0)

def bucket_paths(path='~/train_32x32', outfolder='~/train_32x32/', n=1):
    
    path = join(os.path.dirname(os.path.abspath(__file__)),path)
    outfolder = join(os.path.dirname(os.path.abspath(__file__)),outfolder)
    
    assert isinstance(path, str), "Expected a string input for the path"
    assert os.path.exists(path), "Input path doesn't exist"

    # make list of files
    files = [join(path,f) for f in listdir(path) if isfile(join(path, f))]
    print('Number of valid images is:', len(files))
    
    images = [_process_path(path) for path in files]
    kmeans = bucket([x for x in images if x is not None], n)
    files = [files[i] for i in range(len(files)) if images[i] is not None]
    groups = {}
    print(kmeans.labels_)
    for i in range(len(kmeans.labels_)):
        if kmeans.labels_[i] not in groups.keys():
            groups[kmeans.labels_[i]] = []
            groups[kmeans.labels_[i]].append(files[i])
        else:
            groups[kmeans.labels_[i]].append(files[i])
    print("clustering complete")
    
    # check all images for correct shapes etc. and dump them into
    for cluster in groups.keys():
        clust_files = groups[cluster]
        imgs = []
        for i in tqdm(range(len(clust_files))):
            img = cv2.imread(join(path, clust_files[i]))
            img = img.astype('uint8')
            assert img.shape == (32, 32, 3)
            assert np.max(img) <= 255
            assert np.min(img) >= 0
            assert img.dtype == 'uint8'
            assert isinstance(img, np.ndarray)
            imgs.append(img)
        resolution_x, resolution_y = img.shape[0], img.shape[1]
        imgs = np.asarray(imgs).astype('uint8')
        assert imgs.shape[1:] == (resolution_x, resolution_y, 3)
        assert np.max(imgs) <= 255
        assert np.min(imgs) >= 0
        print('Total number of images is:', imgs.shape[0])
        print('All assertions done, dumping into npy file')
        outfile = join(outfolder, "bucket"+str(cluster)+".npy")
        if not os.path.isdir(os.path.dirname(os.path.abspath(outfile))):
            os.makedirs(os.path.dirname(os.path.abspath(outfile)))
        np.save(outfile, imgs)
    

if __name__ == '__main__':
    number_of_buckets = 5
    bucket_paths(path='data/train_32x32',outfolder='data/bucket_imagenet/train',n=number_of_buckets)
    bucket_paths(path='data/valid_32x32',outfolder='data/bucket_imagenet/valid',n=number_of_buckets)