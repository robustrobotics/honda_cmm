# Test using python-register
# Estimates a warp field between two images using thin plate splines

import numpy as np
import yaml
import matplotlib.pyplot as plt

from pythonregister.imreg import model, register
from pythonregister.imreg.samplers import sampler
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

image = register.RegisterData(im1, features=yaml.load(open('image_features.yaml')))
template = register.RegisterData(im2, features=yaml.load(open('template_features.yaml')))

# Form the feature registrator
feature = register.FeatureRegister(
    model=model.ThinPlateSpline,
    sampler=sampler.Spline,
    )

# Perform the registration
p, warp, img, error = feature.register(image, template)
print("Thin-plate Spline kernel error: ", error)

fig = plt.figure();
plt.subplot(131); plt.imshow(im1)
plt.subplot(132); plt.imshow(im2)
plt.subplot(133); plt.imshow(img)
plt.show()
