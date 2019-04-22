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


def frame_to_array(frame, dtype):
    # PyBullet has a flat list with RGBA values.
    if dtype == 'rgba':
        img = np.array(frame, dtype='uint8').reshape(200, 200, 4)
    elif dtype == 'depth':
        img = (np.array(frame)-0.9)/0.1 *255
        img = np.array(img, dtype='uint8').reshape(200, 200, 1)
        return img
    # PyBullet has RGB but opencv user BGR.
    tmp_red = img[:,:,0].tolist()
    img[:,:,0] = img[:,:,2]
    img[:,:,2] = np.array(tmp_red)
    return img[:,:,:3]


def color_segments(cc):
    colors = {}
    for i in range(cc[0]):
        colors[i] = [np.random.randint(0, 255), 
                     np.random.randint(0, 255), 
                     np.random.randint(0, 255)]
    img = np.zeros((cc[1].shape[0], cc[1].shape[1], 3), np.uint8)
    for ix in range(cc[1].shape[0]):
        for jx in range(cc[1].shape[1]):
            img[ix, jx, :] = colors[cc[1][ix, jx]]

    cv2.imshow('segments', img)


def depth_segmentation(frame, color_frame, diff=10):
    cv2.imshow('depth', frame)

    # Step 1: Find edges in the depth image.
    edges = cv2.Canny(image=frame,
                      threshold1=50,
                      threshold2=150,
                      apertureSize=5)
    cv2.imshow('orig_edges', edges)

    # Step 2: Dilate edges to close regions.
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
    edges = cv2.erode(edges, np.ones((5, 5), np.uint8), iterations=1)
    cv2.imshow('raw_edges', edges)

    # Step 3: Flood fill each region with a new color.
    # im2 = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    im2 = np.ones((edges.shape[0], edges.shape[1], 3), np.uint8)
    im2[:, :, 0] = edges
    im2[:, :, 1] = edges
    im2[:, :, 2] = edges
    mask = np.zeros((edges.shape[0] + 2, edges.shape[1] + 2), dtype=np.uint8)
    for ix in range(0, edges.shape[0]):
        for jx in range(0, edges.shape[1]):
            if mask[ix + 1, jx + 1] == 0:
                new_color = [np.random.randint(0, 255),
                             int(im2[ix, jx, 1]),
                             int(im2[ix, jx, 2])]
                print(new_color)
                _, im2, mask, _ = cv2.floodFill(image=im2,
                                                mask=mask,
                                                seedPoint=(jx, ix),
                                                newVal=new_color,
                                                loDiff=(diff, diff, diff),
                                                upDiff=(diff, diff, diff),
                                                flags=cv2.FLOODFILL_FIXED_RANGE)
    edges = im2
    cv2.imshow('depth_seg', edges)

    # Step 3: Flood fill each region with a new color.
    hsv = cv2.cvtColor(color_frame, cv2.COLOR_BGR2HSV)
    im2 = np.ones(hsv.shape, np.uint8)
    im2[:, :, 0] = hsv[:, :, 0]
    im2[:, :, 1] = hsv[:, :, 1]
    im2[:, :, 2] = edges[:, :, 0]

    mask = np.zeros((edges.shape[0] + 2, edges.shape[1] + 2), dtype=np.uint8)
    for ix in range(0, edges.shape[0]):
        for jx in range(0, edges.shape[1]):
            if mask[ix + 1, jx + 1] == 0:
                new_color = [np.random.randint(0, 255),
                             int(im2[ix, jx, 1]),
                             int(im2[ix, jx, 2])]
                print(new_color)
                _, im2, mask, _ = cv2.floodFill(image=im2,
                                                mask=mask,
                                                seedPoint=(jx, ix),
                                                newVal=new_color,
                                                loDiff=(diff, diff, diff),
                                                upDiff=(diff, diff, diff),
                                                flags=cv2.FLOODFILL_FIXED_RANGE)



    # Step 4: Show the result.
    cv2.imshow('segmented', im2)
    cv2.waitKey(0)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', required=True, type=str, help='[optflow|color|none]')
    args = parser.parse_args()

    with open('video2.pkl', 'rb') as handle:
        frames = pickle.load(handle)
    print('Loaded %d frames.' % len(frames)) 
    print('Converting images...')
    rgb_frames = [frame_to_array(frame['rgb'], dtype='rgba') for frame in frames][::2]
    depth_frames = [frame_to_array(frame['depth'], dtype='depth') for frame in frames][::2]

    for ix in range(1, len(frames)):
        if args.type == 'optflow':
            get_optical_flow(rgn_frames[ix-1], rgb_frames[ix])
        elif args.type == 'color':
            get_color_segmentation(rgb_frames[ix])
        elif args.type == 'rgb':
            cv2.imshow('video', rgb_frames[ix])
            cv2.waitKey(0)
        elif args.type == 'depth':
            depth_segmentation(depth_frames[ix], rgb_frames[ix])
    cv2.destroyAllWindows()

