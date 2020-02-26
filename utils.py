import json



"""
In this module there are many useful fuctions
"""



def json_to_keypoints(filename):

	"""
    This function takes the json file from openpose
    and returns a concatenated list of all keypoints'''
	"""

	keys = ['pose_keypoints_2d','face_keypoints_2d','hand_left_keypoints_2d','hand_right_keypoints_2d']
    with open(filename) as json_file:
        data = json.load(json_file)
        people_dict = data['people']
        keypoints_dict = people_dict[0]
        keypoints = []
        for key in keys:
            points = keypoints_dict[key]
            for i, point in enumerate(points,1):
            	if i%3 != 0:
            		keypoints.append(point)
    return keypoints
