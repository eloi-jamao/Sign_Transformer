import os
import json
import spacy

root = os.getcwd()

'''Loading keypoints'''
videos_folder = root + '/data/How2Sign_samples 2/How2Sign_samples/openpose_output/json'
videos = os.listdir(videos_folder)
keypoints_files = os.listdir(videos_folder + '/' + videos[0])



def read_json(filename):
    '''This function takes the json file from openpose
    and returns a concatenated list of all keypoints'''
    keys = ['pose_keypoints_2d','face_keypoints_2d','hand_left_keypoints_2d','hand_right_keypoints_2d']
    with open(filename) as json_file:
        data = json.load(json_file)
        people_dict = data['people']
        keypoints_dict = people_dict[0]
        keypoints = []
        for key in keys:
            points = keypoints_dict[key]
            keypoints += points
    return keypoints

for i in range(5):
    frame_keypoints = read_json(videos_folder + '/' + videos[0] +'/' + keypoints_files[i])
    print(f'Frame {i} with {len(frame_keypoints)} keypoints and type {type(frame_keypoints)}')
print(len(keypoints_files))

'''Loading text'''
text = r"This is a one minute video and i don't know what she is saying so i'm just making it up while watching it, also i think all the gestures are very well executed although I don't speak sign language. Tomorrow there will be some sun in the day and some rain in the night."

spacy_en = spacy.load('en')

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]
