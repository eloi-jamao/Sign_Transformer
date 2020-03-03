import json
import os
"""
In this module there are many useful fuctions
"""
def json2keypoints(filename):
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

def kp_padding(kp_array, max_len=100):
	while len(kp_array)<max_len:
		kp_array.append(0)



if __name__ == '__main__':
	
	root = os.getcwd()

	''''Loading keypoints'''
	videos_folder = root + '/data/How2Sign_samples 2/How2Sign_samples/openpose_output/json'
	videos = os.listdir(videos_folder)
	keypoints_files = os.listdir(videos_folder + '/' + videos[0])
	for i in range(2):
		frame_keypoints = json2keypoints(videos_folder + '/' + videos[0] +'/' + keypoints_files[i])
		print(f'Frame {i} with {len(frame_keypoints)} keypoints and type {type(frame_keypoints)}') #Uncomment to print results
