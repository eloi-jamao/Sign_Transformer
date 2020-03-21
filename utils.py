import json
import os
import numpy as np
import csv
import operator
#import seaborn as sns
import matplotlib.pyplot as plt
#import seaborn as sns
"""
In this module there are many useful fuctions
"""


base_dir = './data/keypoints/test'
dirs = os.listdir(base_dir)


# expected number of keypoints
# we can try to reduce the number of face keypoints since it seems that
# face expressions are not crucial for SLT
keypoints_count = {
    "pose_keypoints_2d": 75,
    "face_keypoints_2d": 210,
    "hand_left_keypoints_2d": 63,
    "hand_right_keypoints_2d": 63
}


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



def calculate_max_frames(features_path):
	tte = os.listdir(features_path)
	l = []
	for dir in tte:
		print(dir)
		for dir2 in os.listdir(os.path.join(features_path, dir)):
			lenght = len(os.listdir(os.path.join(features_path, dir, dir2)))
			l.append(lenght)

	return max(l), len(l)


def show_frames_stats():
    """
    Computes some statistics on input sequences
    """
    frames_per_sentence = []
    print(f"Number of sentences: {len(dirs)}")  # 642
    for dir in dirs:
        frames_per_sentence.append(len(os.listdir(f"{base_dir}/{dir}")))
        #print(frames_per_sentence[-1])
    print("Statistics:\n")
    print(f"Sum frames {sum(frames_per_sentence)}")  # consistent with paper
    print(f"Max length {max(frames_per_sentence)}")
    print(f"Min length {min(frames_per_sentence)}")
    print(f"Mean length {np.mean(frames_per_sentence)}")
    print(f"Median length {np.percentile(frames_per_sentence, 50)}")
    print(f"Std of length {np.std(frames_per_sentence)}")
    sns.distplot(frames_per_sentence)
    plt.title("Input length distribution")
    plt.show()


def check_files():
    """
    Assert that all the files contain expected number of keypoints
    """
    valid_keys = list(keypoints_count.keys())
    for dir in dirs:
        for file in os.listdir(f"{base_dir}/{dir}"):
            #print(f"{base_dir}/{dir}/{file}")
            with open(f"{base_dir}/{dir}/{file}", "r") as f:
                content = json.load(f)["people"][0]
                for k in valid_keys:
                    # assert len(content[k]) == keypoints_count[k]
                    if len(content[k]) != keypoints_count[k]:
                        print(f"{base_dir}/{dir}/{file}")


def confidence_stats():
    """
    Computes the average confidence score per keypoints of single frame
    and shows a distributions of confidence within all frames
    """
    valid_keys = list(keypoints_count.keys())
    valid_keys.remove("pose_keypoints_2d")
    # since in SLT task we don't care about legs' position, we can discard
    # the following keypoints
    invalid_pose_keypoints = np.array([10, 11, 13, 14, 19, 20, 21, 22, 23, 24])
    invalid_pose_indices = (invalid_pose_keypoints + 1) * 3
    avg_conf_scores = []
    for dir in dirs:
        for file in os.listdir(f"{base_dir}/{dir}"):
            with open(f"{base_dir}/{dir}/{file}", "r") as f:
                content = json.load(f)["people"][0]
                conf_scores = []
                for i, elem in enumerate(content["pose_keypoints_2d"], 1):
                    if i not in invalid_pose_indices and i%3 == 0:
                        conf_scores.append(elem)
                for k in valid_keys:
                    for i, elem in enumerate(content[k], 1):
                        if i%3 == 0:
                            conf_scores.append(elem)
                avg_conf_scores.append(np.mean(conf_scores))
    sns.distplot(avg_conf_scores)
    plt.title("Confidence distribution")
    plt.show()


def create_gloss_vocab():
    vocabulary = []
    token_vocab = {}
    num_tokens=0
    annotation_path = "data/annotations/"
    for csv_file in ["dev.csv", "train.csv", "test.csv"]:
        csv_path = annotation_path + csv_file
        print(csv_path)
        with open(csv_path) as file:
            csv_reader = csv.reader(file, delimiter='|')
            next(csv_reader)
            for row in csv_reader:
                for gword in row[-2].split():
                    num_tokens += 1
                    if gword not in vocabulary:
                        #print(gword)
                        vocabulary.append(gword)
                        token_vocab[gword] = 1
                    else:
                        token_vocab[gword] += 1
    vocabulary.sort()
    list_dict = list(token_vocab.items())
    list_dict.sort(key = operator.itemgetter(1), reverse = True)
    #print(list_dict)
    print("number of tokens = ", num_tokens)
    #print(len(vocabulary))
    coverage_list = []
    for i in range(len(list_dict)):
        if list_dict[i][1] > 1:
            coverage_list.append(list_dict[i])
    print(coverage_list)
    print(len(coverage_list))
    print(compute_coverage(coverage_list, num_tokens))



    """
    #create a vocabulary file .txt
    vocabulary_file = annotation_path + "gloss_tokens.txt"
    with open(vocabulary_file, "w") as file:
        for gword, reps in list_dict:
            file.write(gword+ "  "+ str(reps) + "\n")
    """


def compute_coverage(list_vocab, num_tokens):
    count = 0
    for word, tok in list_vocab:
        count += tok
    return count/num_tokens





if __name__ == '__main__':

    create_gloss_vocab()


	#features_path = "data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px"
	#print(calculate_max_frames(features_path))
	#show_frames_stats()  # ok
    #check_files()  # ok
    #confidence_stats()  # based on confidence distribution we can take frames with average confidence > 0.25



    """
	'Loading keypoints'''
	videos_folder = root + '/data/How2Sign_samples 2/How2Sign_samples/openpose_output/json'
	videos = os.listdir(videos_folder)
	keypoints_files = os.listdir(videos_folder + '/' + videos[0])
	for i in range(2):
		frame_keypoints = json2keypoints(videos_folder + '/' + videos[0] +'/' + keypoints_files[i])
		print(f'Frame {i} with {len(frame_keypoints)} keypoints and type {type(frame_keypoints)}') #Uncomment to print results
	"""
