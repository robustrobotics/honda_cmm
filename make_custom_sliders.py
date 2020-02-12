import numpy as np
from gen.generator_busybox import Slider, BusyBox
from utils.setup_pybullet import setup_env

width, height = 0.6, 0.6 # busybox dimensions
x_offset = 0.0 # in (-width/2, width/2)
z_offset = 0.0 # in (-height/2, height/2)
range = 0.3 # in (0.1, 0.5)
angle = np.pi/2 # in (0, np.pi)
axis = (np.cos(angle), np.sin(angle))
color = (1., 0., 0.)
slider = Slider(x_offset, z_offset, range, axis, color)

bb = BusyBox.get_busybox(width, height, [slider])
image_data, gripper = setup_env(bb, False, False, True, show_im=True)
