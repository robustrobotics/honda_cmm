# Test using PyElastix

import pyelastix
import matplotlib.pyplot as plt
import numpy as np
from utils.util import read_from_file, imshow
data = read_from_file('test.pickle')

# Read image data
first_bb = data[0][0]
image_data = first_bb.image_data
w, h, pixels = image_data
im1 = imshow(image_data, show=False)

second_bb = data[1][0]
image_data = second_bb.image_data
w, h, pixels = image_data
im2 = imshow(image_data, show=False)

im1 = im1[:,:,1].astype('float32')
im2 = im2[:,:,1].astype('float32')

params = pyelastix.get_default_params(type='Affine')
params.NumberOfResolutions = 3
print(params)

# Register
# import pdb; pdb.set_trace()
im3, field = pyelastix.register(im1, im2, params, verbose=0)
X = np.arange(0, 116, 1)
Y = np.arange(0, 118, 1)
U, V = field[0], field[1]

# Visualize the result
fig, axes = plt.subplots(4,1);
axes[0].imshow(im1)
axes[1].imshow(im2)
axes[2].imshow(im3)
axes[3].quiver(X,Y,U,V)
plt.show()  # mpl

# print('field: ', field)
# print('field shape: ', field[0].shape, field[1].shape)