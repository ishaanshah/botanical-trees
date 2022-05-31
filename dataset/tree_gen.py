import time
import bpy
import json
import os
import random
import numpy as np
import sys
from pathlib import Path
from math import radians

"""
# TODO
- [x] Randomize some parameters like branch density, height, leaf density
- [x] Generate RBV
- [X] Render with background
- [X] Render mask (0 AA)
"""

leave_prop_map = {
    "Twig Object": "Input_23",
    "Radius": "Input_15",
    "Density": "Input_5",
    "Size": "Input_11",
    "Size Variation": "Input_13",
    "Deviation": "Input_9",
    "Randomness": "Input_17",
    "Flatness": "Input_19",
    "Weight": "Input_21",
}


def generate_rbv(dims, rotation):
    layers = dims[0]
    sectors = dims[1]
    points = []

    object = bpy.data.objects["tree"]
    for point in object.data.vertices:
        points.append(np.array(point.co))
    points = np.array(points)

    # Convert x, y to polar
    polar_points = points.copy()
    polar_points[:, 0] = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
    polar_points[:, 1] = np.arctan2(points[:, 1], points[:, 0]) + np.pi

    # Rotate to account for camera angle
    polar_points[:, 1] = (polar_points[:, 1] + rotation) % (2*np.pi)
    points = polar_points

    height = np.max(points[:, 2])
    layer_height = height / layers
    sector_theta = 2 * np.pi / sectors

    rbv = np.zeros(dims)
    points[:, 2][points[:, 2] < 0] = 0
    # Iterate heightwise
    for i in range(layers):
        start_height = i * layer_height
        end_height = start_height + layer_height
        # Iterate sectorwise
        for j in range(sectors):
            start_angle = j * sector_theta
            end_angle = start_angle + sector_theta
            idx = np.where(
                (points[:, 1] >= start_angle)
                & (points[:, 1] < end_angle)
                & (points[:, 2] >= start_height)
                & (points[:, 2] < end_height)
            )
            if idx[0].size:
                rbv[i][j] = np.max(points[idx[0], 0])

    return rbv


def render_rgb(root, force=False):
    # If map is already rendered skip it
    if not force and os.path.exists(os.path.join(root, "rgb.png")):
        return

    bpy.context.scene.render.filepath = os.path.join(
        os.path.join(root, "rgb.png")
    )
    bpy.ops.render.render(write_still=True)


def render_segmentation(root, force=False):
    # If map is already rendered skip it
    if not force and os.path.exists(os.path.join(root, "segmentation.png")):
        return

    # Change twigs and bark material to their masked versions
    tree = bpy.data.objects["tree"]
    tree.modifiers["leaves"]["Input_23"] = bpy.data.objects["Twig Mask"]
    tree.data.materials[0] = bpy.data.materials["Bark Mask"]

    # Change world material to white
    old_world = bpy.data.scenes["Scene"].world
    bpy.data.scenes["Scene"].world = bpy.data.worlds[f"Mask"]

    # Disable AA
    bpy.data.scenes["Scene"].cycles.filter_width = 0.01

    # Render
    bpy.context.scene.render.filepath = os.path.join(root, "segmentation.png")
    bpy.ops.render.render(write_still=True)

    # Restore world material
    bpy.data.scenes["Scene"].world = old_world

    # Revert twigs and materials to their normal versions
    tree.modifiers["leaves"]["Input_23"] = bpy.data.objects["Twig"]
    tree.data.materials[0] = bpy.data.materials["Bark"]

    # Re-enable AA
    bpy.data.scenes["Scene"].cycles.filter_width = 1.5


# Delete default cube and light
if "Cube" in bpy.data.objects:
    bpy.data.objects.remove(bpy.data.objects["Cube"], do_unlink=True)
if "Light" in bpy.data.objects:
    bpy.data.objects.remove(bpy.data.objects["Light"], do_unlink=True)

# Delete old tree
if "tree" in bpy.data.objects:
    bpy.data.objects.remove(bpy.data.objects["tree"], do_unlink=True)

# Command line arguments
tree_model = sys.argv[-3]
model_count = int(sys.argv[-2])
do_render = int(sys.argv[-1])
# tree_model = "neem"
# model_count = 1

bpy.context.scene.frame_set(1)
bpy.context.scene.render.engine = "CYCLES"
bpy.context.scene.render.resolution_y = 1024
bpy.context.scene.render.resolution_x = 1024
bpy.context.scene.render.film_transparent = True
bpy.context.scene.render.image_settings.color_mode = "RGBA"
bpy.context.scene.render.image_settings.file_format = "PNG"
bpy.context.scene.render.image_settings.compression = 100
bpy.context.scene.cycles.samples = 128
bpy.context.scene.view_settings.view_transform = "Standard"
root = os.path.dirname(bpy.data.filepath)

# Read params
with open(os.path.join(root, "presets", f"{tree_model}.json")) as f:
    params = json.load(f)

camera_params = params["Cameras"]

# Import materials and twigs
if "Bark" not in bpy.data.materials:
    bpy.ops.wm.append(
        directory=os.path.join(
            root, "Grammar-Trees", "library", f"{tree_model}.blend", "Material"
        ),
        filename="Bark",
    )

if "Bark Mask" not in bpy.data.materials:
    bpy.ops.wm.append(
        directory=os.path.join(
            root, "Grammar-Trees", "library", "common.blend", "Material"
        ),
        filename="Bark Mask",
    )

if "Mask" not in bpy.data.worlds:
    bpy.ops.wm.append(
        directory=os.path.join(
            root, "Grammar-Trees", "library", "common.blend", "World"
        ),
        filename="Mask",
    )

for i in range(5):
    if f"HDR_{i+1}" not in bpy.data.worlds:
        bpy.ops.wm.append(
            directory=os.path.join(
                root, "Grammar-Trees", "library", "common.blend", "World"
            ),
            filename=f"HDR_{i+1}",
        )

if "Twig" not in bpy.data.objects:
    bpy.ops.wm.append(
        directory=os.path.join(
            root, "Grammar-Trees", "library", f"{tree_model}.blend", "Object"
        ),
        filename="Twig",
    )

if "Twig Mask" not in bpy.data.objects:
    bpy.ops.wm.append(
        directory=os.path.join(
            root, "Grammar-Trees", "library", f"{tree_model}.blend", "Object"
        ),
        filename="Twig Mask",
    )

# Import node group for creating the tree
if not "Tree" in bpy.data.node_groups:
    bpy.ops.wm.append(
        directory=os.path.join(
            root, "Grammar-Trees", "library", f"{tree_model}.blend", "NodeTree"
        ),
        filename="Tree",
    )

t0 = time.time()
cnt_views = 0
cnt_trees = 0
for i in range(model_count):
    # Generate the tree
    nodes = bpy.data.node_groups["Tree"].nodes
    random.seed(i)

    # Set background
    bpy.data.scenes["Scene"].world = bpy.data.worlds[
        f"HDR_{random.randint(1, 5)}"
    ]

    # Set seed for determinism
    for node in nodes:
        if "Seed" in node.inputs:
            node.inputs["Seed"].property_value = i

    # Set trunk properties
    if "Trunk" in params:
        for param, value in params["Trunk"].items():
            if isinstance(value, list):
                bpy.data.node_groups["Tree"].nodes["Trunk"].inputs[
                    param
                ].property_value = random.uniform(*value)
            else:
                bpy.data.node_groups["Tree"].nodes["Trunk"].inputs[
                    param
                ].property_value = value

    # Set branch properties
    if "Branches" in params:
        for param, value in params["Branches"].items():
            if isinstance(value, list):
                bpy.data.node_groups["Tree"].nodes["Branches"].inputs[
                    param
                ].property_value = random.uniform(*value)
            else:
                bpy.data.node_groups["Tree"].nodes["Branches"].inputs[
                    param
                ].property_value = value

    # Generate the tree
    bpy.ops.mtree.node_function(
        node_tree_name="Tree",
        node_name="Tree Mesher",
        function_name="build_tree",
    )

    # Generate leaves
    bpy.ops.mtree.add_leaves(object_id="tree")

    tree = bpy.data.objects["tree"]

    # Set twig object
    tree.modifiers["leaves"]["Input_23"] = bpy.data.objects["Twig"]

    # Assign bark material
    if tree.data.materials:
        tree.data.materials[0] = bpy.data.materials["Bark"]
    else:
        tree.data.materials.append(bpy.data.materials["Bark"])

    # Set other leaf parameters
    for param, value in params["Leaves"].items():
        if isinstance(value, list):
            bpy.data.objects["tree"].modifiers["leaves"][
                leave_prop_map[param]
            ] = random.uniform(*value)
        else:
            bpy.data.objects["tree"].modifiers["leaves"][
                leave_prop_map[param]
            ] = value


    # Create directory to store files of single tree model
    for j in range(len(camera_params)):
        # Set camera position
        bpy.data.objects["Camera"].location.x = camera_params[j][0][0]
        bpy.data.objects["Camera"].location.y = camera_params[j][0][1]
        bpy.data.objects["Camera"].location.z = camera_params[j][0][2]

        # Set camera rotation
        bpy.data.objects["Camera"].rotation_euler.x = radians(
            camera_params[j][1][0]
        )
        bpy.data.objects["Camera"].rotation_euler.y = radians(
            camera_params[j][1][1]
        )
        bpy.data.objects["Camera"].rotation_euler.z = radians(
            camera_params[j][1][2]
        )

        # Generate RBV
        rbv = generate_rbv((8, 8), radians(camera_params[j][1][2]))

        data_root = os.path.join(root, "dataset", tree_model, str(i), str(j))
        Path(data_root).mkdir(parents=True, exist_ok=True)

        info = {
            "species": tree_model,
            "rbv": rbv.tolist(),
            "camera_params": camera_params[j],
        }

        # Export information (species, rbv, camera parameters)
        with open(os.path.join(data_root, "info.json"), "w") as f:
            json.dump(info, f, indent=2)

        # Export renders
        if do_render:
            # render_rgb(data_root, force=True)
            # render_segmentation(data_root, force=True)

            render_rgb(data_root)
            render_segmentation(data_root)

        cnt_views += 1
        print(f"{cnt_views}/{model_count*len(camera_params)}", end="\r")

    cnt_trees += 1

print(
    f"Generated {cnt_trees} trees and {cnt_views} renders. "
    f"Took {time.time() - t0}"
)
