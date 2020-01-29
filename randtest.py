import numpy as np 

im1 = np.array([[[ 0.,  1.,  3.,  0.],
        [ 0.,  2.,  0.,  6.],
        [ 8.,  0.,  7.,  0.]],

       [[ 0.,  9.,  0.,  3.],
        [ 11.,  2.,  4.,  0.],
        [ 0.,  6.,  5.,  1.]]])
print(im1)
im1 = im1[:,:,1].astype('float32')
print(im1)