import os
import json
import numpy as np

root = os.getcwd()

'''Loading keypoints'''
videos_folder = os.path.join(root, "data", "How2Sign_samples", "openpose_output", "json")
videos = os.listdir(videos_folder)
video_folder = os.path.join(videos_folder, videos[0])
keypoints_files = os.listdir(video_folder)


def read_json_2_openpose1dframe(filename):
    '''This function takes the json file from openpose
    and returns a concatenated list of all keypoints'''
    # keys = ['pose_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d']
    with open(filename) as json_file:
        data = json.load(json_file)
        people_dict = data['people']
        keypoints_dict = people_dict[0]

        keypoints_x = []
        keypoints_y = []
        keypoints_c = []
        keypoints = keypoints_dict['pose_keypoints_2d'][:8*3]
        keypoints += keypoints_dict['hand_left_keypoints_2d']
        keypoints += keypoints_dict['hand_right_keypoints_2d']
        for i, point in enumerate(keypoints):
            j = i % 3
            if j == 0:
                keypoints_x.append(point)
            elif j == 1:
                keypoints_y.append(point)
            elif j == 2:
                keypoints_c.append(point)
    return [keypoints_x, keypoints_y, keypoints_c]

//TODO

keypoints = [ [ None for y in range( 50) ]
             for x in range( 3 ) ]

for file in keypoints_files:
    json_file = os.path.join(video_folder, file)
    frame_keypoints = read_json(json_file)
    keypoints = np.stack((keypoints, frame_keypoints), axis=0)
    # print(f'Frame {i} with {len(frame_keypoints)} keypoints and type {type(frame_keypoints)}')
print(len(keypoints_files))
