import cv2
import pickle
import numpy as np
import time
import sys
import argparse

hsv = np.zeros((200, 200, 3), dtype='uint8')

hsv[...,1] = 255
def get_optical_flow(frame1, frame2):
    prev = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 5, 10, 3, 5, 1.1, 0)

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    # hsv[...,0] = ang*180./np.pi/2.
    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
    

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    cv2.imshow('flow', binary)
    cv2.waitKey(0)

def get_color_segmentation(frame, diff=50):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    im2 = cv2.pyrMeanShiftFiltering(frame, 25, 50)

    mask = np.zeros((im2.shape[0]+2, im2.shape[1]+2), dtype=np.uint8)
    for ix in range(0, im2.shape[0]):
        for jx in range(0, im2.shape[1]):
            if mask[ix+1,jx+1] == 0 or True:
                _, im2, mask, _ = cv2.floodFill(image=im2, 
                                                mask=mask, 
                                                seedPoint=(jx, ix), 
                                                newVal=im2[ix,jx,:].tolist(), 
                                                loDiff=(diff, diff, diff), 
                                                upDiff=(diff, diff, diff),
                                                flags=cv2.FLOODFILL_FIXED_RANGE) 
    cv2.imshow('color', im2)
    cv2.waitKey(0)


def frame_to_array(frame):
    # PyBullet has a flat list with RGBA values.
    img = np.array(frame, dtype='uint8').reshape(200, 200, 4)
    # PyBullet has RGB but opencv user BGR.
    tmp_red = img[:,:,0].tolist()
    img[:,:,0] = img[:,:,2]
    img[:,:,2] = np.array(tmp_red)
    return img[:,:,:3]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', required=True, type=str, help='[optflow|color|none]')
    args = parser.parse_args()

    with open('video.pkl', 'rb') as handle:
        frames = pickle.load(handle)
    print('Loaded %d frames.' % len(frames)) 
    print('Converting images...')
    
    frames = [frame_to_array(frame) for frame in frames][::2]
    for ix in range(1, len(frames)):
        if args.type == 'optflow':
            get_optical_flow(frames[ix-1], frames[ix])
        elif args.type == 'color':
            get_color_segmentation(frames[ix])
        elif args.type == 'none':
            cv2.imshow('video', frames[ix])
            cv2.waitKey(0)
    cv2.destroyAllWindows()

