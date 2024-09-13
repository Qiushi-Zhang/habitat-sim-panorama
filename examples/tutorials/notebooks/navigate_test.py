import math
import os
import random

import git
import imageio
import magnum as mn
import numpy as np

import math
import os
import random

import git
import imageio
import magnum as mn
import numpy as np
import cv2
import stitch
import utils
import timeit
import argparse
import os

# %matplotlib inline
from matplotlib import pyplot as plt

# function to display the topdown map
from PIL import Image

import habitat_sim
from habitat_sim.utils import common as utils
from habitat_sim.utils import viz_utils as vut

# %matplotlib inline
from matplotlib import pyplot as plt

# function to display the topdown map
from PIL import Image

import habitat_sim
from habitat_sim.utils import common as utils
from habitat_sim.utils import viz_utils as vut

repo = git.Repo(".", search_parent_directories=True)
dir_path = repo.working_tree_dir
data_path = os.path.join(dir_path, "data")
print(f"data_path = {data_path}")
# @markdown Optionally configure the save path for video output:
output_directory = os.path.join(
    dir_path, "examples/tutorials/nav_output/"
)  # @param {type:"string"}
output_path = os.path.join(dir_path, output_directory)
if not os.path.exists(output_path):
    os.mkdir(output_path)





# @title Define Observation Display Utility Function { display-mode: "form" }

# @markdown A convenient function that displays sensor observations with matplotlib.

# @markdown (double click to see the code)


# Change to do something like this maybe: https://stackoverflow.com/a/41432704
def display_sample(rgb_obs, semantic_obs=np.array([]), depth_obs=np.array([])):
    from habitat_sim.utils.common import d3_40_colors_rgb

    rgb_img = Image.fromarray(rgb_obs, mode="RGBA")

    arr = [rgb_img]
    titles = ["rgb"]
    if semantic_obs.size != 0:
        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
        arr.append(semantic_img)
        titles.append("semantic")

    if depth_obs.size != 0:
        depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")
        arr.append(depth_img)
        titles.append("depth")

    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        plt.imshow(data)
    plt.show(block=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.add_argument("--no-make-video", dest="make_video", action="store_false")
    parser.set_defaults(show_video=True, make_video=True)
    args, _ = parser.parse_known_args()
    show_video = args.display
    display = args.display
    do_make_video = args.make_video
else:
    show_video = False
    do_make_video = False
    display = False

# import the maps module alone for topdown mapping
if display:
    from habitat.utils.visualizations import maps

test_scene = os.path.join(
    data_path, "scene_datasets/mp3d_example/17DRP5sb8fy/17DRP5sb8fy.glb"
)

sim_settings = {
    "scene": test_scene,  # Scene path
    "default_agent": 0,  # Index of the default agent
    "sensor_height": 1.5,  # Height of sensors in meters, relative to the agent
    "width": 256,  # Spatial resolution of the observations
    "height": 256,
}

def make_simple_cfg(settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]

    # agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    # In the 1st example, we attach only one sensor,
    # a RGB visual sensor, to the agent
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]

    agent_cfg.sensor_specifications = [rgb_sensor_spec]

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


cfg = make_simple_cfg(sim_settings)

try:  # Needed to handle out of order cell run in Jupyter
    sim.close()
except NameError:
    pass
sim = habitat_sim.Simulator(cfg)


agent = sim.initialize_agent(sim_settings["default_agent"])

# Set agent state
agent_state = habitat_sim.AgentState()
agent_state.position = np.array([-0.6, 0.0, 0.0])  # in world space
agent.set_state(agent_state)

# Get agent state
agent_state = agent.get_state()
print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)

# obtain the default, discrete actions that an agent can perform
# default action space contains 3 actions: move_forward, turn_left, and turn_right
action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())
print("Discrete action space: ", action_names)


def navigateAndSee(action=""):
    if action in action_names:
        observations = sim.step(action)
        print("action: ", action)
        if display:
            display_sample(observations["color_sensor"])
            
def wrap_angle(angle):
    """Wrap the angle to be from -pi to pi."""
    return (angle + np.pi) % (2 * np.pi) - np.pi
def show_panorama(agent_id,agent_state):
    obs_list = []
    X = agent_state.position[0]
    Y = agent_state.position[1]
    Z = agent_state.position[2]
    original_state = agent_state 
    for rot in [0,40,80,120,160,200,240,280,320]:
        yaw = habitat_sim.utils.common.quat_to_angle_axis (agent_state.rotation) 
        print(yaw)
        heading_angle = rot / 180 * np.pi
        heading_angle = wrap_angle(heading_angle + yaw[0])

        agent_rot = habitat_sim.utils.common.quat_from_angle_axis(
            heading_angle, habitat_sim.geo.GRAVITY)
        
        agent_state.rotation = agent_rot 
        agent.set_state(agent_state)
        obs = sim.get_sensor_observations(0)
        obs_list.append(obs)
    agent_state = original_state 
    agent.set_state(original_state)

    
# ======================= stitch the views to form a panorama
    rgb_lst, depth_lst, sseg_lst = [], [], []

    for idx, obs in enumerate(obs_list):
    # load rgb image, depth and sseg
        rgb_img = obs['color_sensor'][:,:,:3]
        # depth_img = obs['depth_sensor'][:, :, 0]

        img_dir = "/home/qiushi/habitat-sim/examples/tutorials/notebooks/"
        depth_path_png = os.path.join(img_dir, f'depth_image_{idx+1}.jpg')
        cv2.imwrite(depth_path_png, rgb_img)
        

        

        rgb_lst.append(rgb_img)
        # depth_lst.append(depth_img)
        # sseg_lst.append(sseg_img)
        print("shape of rgb img is", rgb_img.shape)

        panorama_rgb = np.concatenate(rgb_lst, axis=1)
        # panorama_depth = np.concatenate(depth_lst, axis=1)
        # panorama_sseg = np.concatenate(sseg_lst, axis=1)


    # depth_stitch = stitch.multiStitching(depth_lst)
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 8))
    ax[0].imshow(panorama_rgb)
    ax[0].get_xaxis().set_visible(False)
    ax[0].get_yaxis().set_visible(False)
    ax[0].set_title("rgb")
    rgb_stitch = stitch.multiStitching(rgb_lst)
    
    

    plt.show()


action = "turn_right"
navigateAndSee(action)

agent_state = agent.get_state()
print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)
show_panorama(0,agent_state)

action = "turn_right"
navigateAndSee(action)

action = "move_forward"
navigateAndSee(action)

action = "turn_left"
navigateAndSee(action)





# test_scene = os.path.join(
#     data_path, "scene_datasets/mp3d_example/17DRP5sb8fy/17DRP5sb8fy.glb"
# )
# mp3d_scene_dataset = os.path.join(
#     data_path, "scene_datasets/mp3d_example/mp3d.scene_dataset_config.json"
# )

# rgb_sensor = True  # @param {type:"boolean"}
# depth_sensor = True  # @param {type:"boolean"}
# semantic_sensor = True  # @param {type:"boolean"}

# sim_settings = {
#     "width": 256,  # Spatial resolution of the observations
#     "height": 256,
#     "scene": test_scene,  # Scene path
#     "scene_dataset": mp3d_scene_dataset,  # the scene dataset configuration files
#     "default_agent": 0,
#     "sensor_height": 1.5,  # Height of sensors in meters
#     "color_sensor": rgb_sensor,  # RGB sensor
#     "depth_sensor": depth_sensor,  # Depth sensor
#     "semantic_sensor": semantic_sensor,  # Semantic sensor
#     "seed": 1,  # used in the random navigation
#     "enable_physics": False,  # kinematics only
# }


print("successfully run ")
