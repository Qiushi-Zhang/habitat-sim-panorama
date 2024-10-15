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
from habitat.utils.visualizations import maps
from stitching import Stitcher 




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


rgb_dir = os.path.join(dir_path, "examples/tutorials/rgb_output")
if not os.path.exists(rgb_dir):
    os.mkdir(rgb_dir)
depth_dir = os.path.join(dir_path, "examples/tutorials/depth_output")
if not os.path.exists(depth_dir):
    os.mkdir(depth_dir)
pano_rgb_dir = os.path.join(dir_path, "examples/tutorials/pano_rgb_output")
if not os.path.exists(pano_rgb_dir):
    os.mkdir(pano_rgb_dir)
pano_depth_dir = os.path.join(dir_path, "examples/tutorials/pano_depth_output")
if not os.path.exists(pano_depth_dir):
    os.mkdir(pano_depth_dir)
    



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


def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = settings["scene"]
    # sim_cfg.scene_dataset_config_file = settings["scene_dataset"]
    sim_cfg.enable_physics = settings["enable_physics"]

    # Note: all sensors must have the same resolution
    sensor_specs = []

    color_sensor_spec = habitat_sim.CameraSensorSpec()
    color_sensor_spec.uuid = "color_sensor"
    color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor_spec.resolution = [settings["height"], settings["width"]]
    color_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(color_sensor_spec)

    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(depth_sensor_spec)

    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(semantic_sensor_spec)

    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

test_scene = os.path.join(
    data_path, "scene_datasets/mp3d_example/17DRP5sb8fy/17DRP5sb8fy.glb"
)
replica_scene_dataset = "/home/qiushi/habitat-sim/data/versioned_data/mp3d_example_scene_1.1/mp3d.scene_dataset_config.json"

rgb_sensor = True  # @param {type:"boolean"}
depth_sensor = True  # @param {type:"boolean"}
semantic_sensor = True  # @param {type:"boolean"}
display = True

sim_settings = {
    "width": 1280,  # Spatial resolution of the observations
    "height": 720,
    "scene": test_scene,  # Scene path
    "scene_dataset": replica_scene_dataset, 
    "default_agent": 0,
    "sensor_height": 1.5,  # Height of sensors in meters
    "color_sensor": rgb_sensor,  # RGB sensor
    "depth_sensor": depth_sensor,  # Depth sensor
    "semantic_sensor": semantic_sensor,  # Semantic sensor
    "seed": 1,  # used in the random navigation
    "enable_physics": False,  # kinematics only
}

def display_map(topdown_map, key_points=None):
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    plt.imshow(topdown_map)
    # plot points on map
    if key_points is not None:
        for point in key_points:
            plt.plot(point[0], point[1], marker="o", markersize=10, alpha=0.8)
    plt.show(block=False)
    plt.savefig("../../tutorials/nav_output/map.jpg")

def wrap_angle(angle):
    """Wrap the angle to be from -pi to pi."""
    return (angle + np.pi) % (2 * np.pi) - np.pi
def generate_panorama(i,agent,sim,agent_state):
    obs_list = []
    stitcher = Stitcher()
    # stitcher = Stitcher(detector="sift", confidence_threshold=0.4,crop=False)
    X = agent_state.position[0]
    Y = agent_state.position[1]
    Z = agent_state.position[2]
    original_state = agent_state 
    rgb_pathes = []
    for rot in [0,30,60,90,120,150,180,210,240,270,300,330,360]:
        yaw = habitat_sim.utils.common.quat_to_angle_axis (agent_state.rotation) 
        heading_angle = rot / 180 * np.pi
        heading_angle = wrap_angle(heading_angle + yaw[0])

        agent_rot = habitat_sim.utils.common.quat_from_angle_axis(
            heading_angle, habitat_sim.geo.GRAVITY)
        
        agent_state.rotation = agent_rot 
        agent.set_state(agent_state)
        observations = sim.get_sensor_observations()
        rgb = observations["color_sensor"]
        semantic = observations["semantic_sensor"]
        depth = observations["depth_sensor"]
                        
        rgb_file = Image.fromarray(rgb)
        rgb_file = rgb_file.convert("RGB")
        rgb_path = os.path.join (pano_rgb_dir, "path_"+str(i)+str(rot)+"degree"+"rgb.jpg")
        rgb_file.save(rgb_path)
        rgb_pathes.append(rgb_path)
        depth_file = Image.fromarray(depth)
        depth_file = depth_file.convert("L")
        depth_path = os.path.join(pano_depth_dir, "path"+str(i)+str(rot)+"degree"+"depth.png")
        depth_file.save(depth_path)
    rgb_pathes.sort()
    panorama = stitcher.stitch(rgb_pathes)
    cv2.imwrite(os.path.join(pano_rgb_dir,"path"+str(i)+"pano_rgb.jpg"),panorama)
    
    




cfg = make_cfg(sim_settings)
try:  # Got to make initialization idiot proof
    sim.close()
except NameError:
    pass
sim = habitat_sim.Simulator(cfg)

# # the navmesh can also be explicitly loaded
# sim.pathfinder.load_nav_mesh(
#     "./data/scene_datasets/habitat-test-scenes/apartment_1.navmesh"
# )

agent = sim.initialize_agent(sim_settings["default_agent"])

# Set agent state
agent_state = habitat_sim.AgentState()
agent_state.position = np.array([-0.6, 0.0, 0.0])  # in world space
agent.set_state(agent_state)

# Get agent state
agent_state = agent.get_state()
print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)

navmesh_settings = habitat_sim.NavMeshSettings()

# @markdown Choose Habitat-sim defaults (e.g. for point-nav tasks), or custom settings.
use_custom_settings = False  # @param {type:"boolean"}
sim.navmesh_visualization = True  # @param {type:"boolean"}
navmesh_settings.set_defaults()
navmesh_success = sim.recompute_navmesh(sim.pathfinder, navmesh_settings)


if not sim.pathfinder.is_loaded:
    print("Pathfinder not initialized, aborting.")
else:
    seed = 5  # @param {type:"integer"}
    sim.pathfinder.seed(seed)

    # fmt off
    # @markdown 1. Sample valid points on the NavMesh for agent spawn location and pathfinding goal.
    # fmt on
    init_point = sim.pathfinder.get_random_navigable_point()
    final_point = sim.pathfinder.get_random_navigable_point()
    total_path_points = []
    for i in range(10):
        init_point = final_point 
        final_point = sim.pathfinder.get_random_navigable_point()
        path = habitat_sim.ShortestPath()
        path.requested_start = init_point
        path.requested_end = final_point
        found_path = sim.pathfinder.find_path(path)
        geodesic_distance = path.geodesic_distance
        print("Location of Source Point :"+str(init_point))
        print("Location of Target Point :"+str(final_point))
        path_points = path.points 
        total_path_points.extend(path_points)
        if found_path:
            meters_per_pixel = 0.025
            scene_bb = sim.get_active_scene_graph().get_root_node().cumulative_bb
            height = scene_bb.y().min
        

            # @markdown 4. (optional) Place agent and render images at trajectory points (if found).
            display_path_agent_renders = True  # @param{type:"boolean"}
            if display_path_agent_renders:
                print("Rendering observations at path points:")
                tangent = path_points[1] - path_points[0]
                agent_state = habitat_sim.AgentState()
                for ix, point in enumerate(path_points):
                    if ix < len(path_points) - 1:
                        tangent = path_points[ix + 1] - point
                        agent_state.position = point
                        tangent_orientation_matrix = mn.Matrix4.look_at(
                            point, point + tangent, np.array([0, 1.0, 0])
                        )
                        tangent_orientation_q = mn.Quaternion.from_matrix(
                            tangent_orientation_matrix.rotation()
                        )
                        agent_state.rotation = utils.quat_from_magnum(tangent_orientation_q)
                        agent.set_state(agent_state)



                        observations = sim.get_sensor_observations()
                        rgb = observations["color_sensor"]
                        semantic = observations["semantic_sensor"]
                        depth = observations["depth_sensor"]
                        
                        rgb_file = Image.fromarray(rgb)
                        rgb_file = rgb_file.convert("RGB")
                        rgb_path = os.path.join (rgb_dir, "path_"+str(i)+"point"+str(ix)+"rgb.jpg")
                        rgb_file.save(rgb_path)
                        depth_file = Image.fromarray(depth)
                        depth_file = depth_file.convert("L")
                        depth_path = os.path.join(depth_dir, "path"+str(i)+"point"+str(ix)+"depth.png")
                        depth_file.save(depth_path)
                    elif ix == len(path_points)-1: 
                        tangent = path_points[ix] - path_points[ix-1]
                        agent_state.position = point
                        tangent_orientation_matrix = mn.Matrix4.look_at(
                            point, point + tangent, np.array([0, 1.0, 0])
                        )
                        tangent_orientation_q = mn.Quaternion.from_matrix(
                            tangent_orientation_matrix.rotation()
                        )
                        agent_state.rotation = utils.quat_from_magnum(tangent_orientation_q)
                        agent.set_state(agent_state)
                        
                        generate_panorama(i,agent,sim,agent_state)
                        print("generating_panorama at path ",i,"point ",ix+1)

                        

                        

                        if display:
                            # display_sample(rgb, semantic, depth)
                            pass 

    
    # @markdown 3. Display trajectory (if found) on a topdown map of ground floor

if display:
            top_down_map = maps.get_topdown_map(
                sim.pathfinder, height, meters_per_pixel=meters_per_pixel
            )
            recolor_map = np.array(
                [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
            )
            top_down_map = recolor_map[top_down_map]
            grid_dimensions = (top_down_map.shape[0], top_down_map.shape[1])
            # convert world trajectory points to maps module grid points
            print("Total number of path points", len(total_path_points))
            trajectory = [
                maps.to_grid(
                    path_point[2],
                    path_point[0],
                    grid_dimensions,
                    pathfinder=sim.pathfinder,
                )
                for path_point in total_path_points
            ]
            grid_tangent = mn.Vector2(
                trajectory[1][1] - trajectory[0][1], trajectory[1][0] - trajectory[0][0]
            )
            path_initial_tangent = grid_tangent / grid_tangent.length()
            initial_angle = math.atan2(path_initial_tangent[0], path_initial_tangent[1])
            # draw the agent and trajectory on the map
            maps.draw_path(top_down_map, trajectory)
            maps.draw_agent(
                top_down_map, trajectory[0], initial_angle, agent_radius_px=8
            )
            print("\nDisplay the map with agent and path overlay:")
            display_map(top_down_map)

