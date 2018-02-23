import os
import sys

sys.path.append(os.path.dirname(__file__) + "/../")

from scipy.misc import imread

from config import load_config
from nnet import predict
from util import visualize
from dataset.pose_dataset import data_to_input


cfg = load_config("./demo/pose_cfg.yaml")

# Load and setup CNN part detector
sess, inputs, outputs = predict.setup_pose_prediction(cfg)

# Read image from file
#file_name ="demo/ttt.jpg"
file_name = "./demo/girl_back.jpg"
# sex=1 for male, sex=0 for female
sex=1
# if image is front image, bf_flag=0, else 1
bf_flag=1

image = imread(file_name, mode='RGB')

image_batch = data_to_input(image)

# Compute prediction with the CNN
outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
scmap, locref, _ = predict.extract_cnn_output(outputs_np, cfg)

# Extract maximum scoring location from the heatmap, assume 1 person
pose = predict.argmax_pose_predict(scmap, locref, cfg.stride)

# Visualise
visualize.show_heatmaps(image, pose,sex,bf_flag)

#visualize.waitforbuttonpress()
