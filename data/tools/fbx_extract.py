import argparse
import warnings
from pathlib import Path

import bpy
import mathutils
import numpy as np

FBX_OUTPUT_SCALE = 0.01

BODY_JOINT_NAMES = [
    "Hips",
    "Spine",
    "Spine1",
    "Spine2",
    "Neck",
    "Head",
    "HeadTop_End",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "LeftHandThumb1",
    "LeftHandThumb2",
    "LeftHandThumb3",
    "LeftHandThumb4",
    "LeftHandIndex1",
    "LeftHandIndex2",
    "LeftHandIndex3",
    "LeftHandIndex4",
    "LeftHandMiddle1",
    "LeftHandMiddle2",
    "LeftHandMiddle3",
    "LeftHandMiddle4",
    "LeftHandRing1",
    "LeftHandRing2",
    "LeftHandRing3",
    "LeftHandRing4",
    "LeftHandPinky1",
    "LeftHandPinky2",
    "LeftHandPinky3",
    "LeftHandPinky4",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "RightHandThumb1",
    "RightHandThumb2",
    "RightHandThumb3",
    "RightHandThumb4",
    "RightHandIndex1",
    "RightHandIndex2",
    "RightHandIndex3",
    "RightHandIndex4",
    "RightHandMiddle1",
    "RightHandMiddle2",
    "RightHandMiddle3",
    "RightHandMiddle4",
    "RightHandRing1",
    "RightHandRing2",
    "RightHandRing3",
    "RightHandRing4",
    "RightHandPinky1",
    "RightHandPinky2",
    "RightHandPinky3",
    "RightHandPinky4",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
    "RightToe_End",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
    "LeftToe_End",
]

JOINT_CONNECTOR = "--"


def delete_all():
    object_types = [
        "MESH",
        "CURVE",
        "SURFACE",
        "META",
        "FONT",
        "HAIR",
        "POINTCLOUD",
        "VOLUME",
        "GPENCIL",
        "ARMATURE",
        "LATTICE",
        "EMPTY",
        "LIGHT",
        "LIGHT_PROBE",
        "CAMERA",
        "SPEAKER",
    ]
    for o in bpy.context.scene.objects:
        for i in object_types:
            if o.type == i:
                o.select_set(False)
            else:
                o.select_set(True)
    bpy.ops.object.delete()


def deselect_all():
    bpy.ops.object.select_all(action="DESELECT")


def get_armature_and_meshes():
    armature = bpy.data.objects["Armature"]
    meshes = [children for children in armature.children if children.type == "MESH"]
    return armature, meshes


def load_fbx(fbx_path, prefix=None):
    delete_all()
    bpy.ops.import_scene.fbx(filepath=str(fbx_path.with_suffix(".fbx")))
    if not prefix:
        _, meshes = get_armature_and_meshes()
        prefix = meshes[0].vertex_groups[0].name.split(":", maxsplit=1)[0] + ":"

    return int(bpy.data.actions[0].frame_range[1]) + 1, prefix


def extract_bone_names(prefix, filter_vertex_groups=False):
    armature, meshes = get_armature_and_meshes()

    bone_names_list = []

    def add_bone_name(bone):
        bone_name = bone.name[len(prefix) :]
        if filter_vertex_groups or (
            (bone_name in BODY_JOINT_NAMES)
            or (
                JOINT_CONNECTOR in bone_name
                and bone_name.split(JOINT_CONNECTOR, maxsplit=1)[0] in BODY_JOINT_NAMES
            )
        ):
            bone_names_list.append(bone_name)
            children = bone.children
            for child in children:
                add_bone_name(child)

    for bone in armature.data.bones:
        if bone.parent is None:
            add_bone_name(bone)

    full_bone_names = {k: i for i, k in enumerate(bone_names_list)}

    if filter_vertex_groups:
        vertex_groups = set(
            sum(
                [
                    [
                        vertex_group.name[len(prefix) :]
                        for vertex_group in mesh.vertex_groups.values()
                    ]
                    for mesh in meshes
                ],
                start=[],
            )
        )
        bone_names_list = [
            bone_name
            for bone_name in bone_names_list
            if (
                (bone_name in vertex_groups)
                and (
                    (bone_name in BODY_JOINT_NAMES)
                    or (
                        JOINT_CONNECTOR in bone_name
                        and bone_name.split(JOINT_CONNECTOR, maxsplit=1)[0]
                        in BODY_JOINT_NAMES
                    )
                )
            )
            or len(
                [
                    child
                    for child in armature.data.bones[
                        prefix + bone_name
                    ].children_recursive
                    if child.name[len(prefix) :] in BODY_JOINT_NAMES
                ]
            )
            > 0
        ]

    bone_names = {k: i for i, k in enumerate(bone_names_list)}

    return bone_names, full_bone_names


def extract_pose_matrices(frame_count, bone_names, prefix):
    armature, _ = get_armature_and_meshes()

    bone_count = len(bone_names)
    matrices = [None] * frame_count

    for i in range(frame_count):
        bpy.context.scene.frame_set(i)

        matrices[i] = [None] * bone_count

        for bone_name, j in bone_names.items():
            bone = armature.pose.bones[prefix + bone_name]
            matrices[i][j] = bone.matrix.copy()

    return matrices


def get_mid_child(parent, children, bone_names, heads, tails):
    if len(children) == 0 or parent is None:
        return

    dist = np.zeros(len(children))
    parent_tail = tails[bone_names[parent]]
    for i, child in enumerate(children):
        dist[i] = np.linalg.norm(heads[bone_names[child]] - parent_tail)
    return children[np.argmin(dist)]


def extract_rest_position(bone_names, prefix):
    armature, _ = get_armature_and_meshes()

    bone_count = len(bone_names)
    heads = np.zeros((bone_count, 3))
    tails = np.zeros_like(heads)
    rolls = np.zeros((bone_count, 1))

    deselect_all()

    prev_mode = armature.mode
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="EDIT")

    for bone_name, j in bone_names.items():
        bone = armature.data.edit_bones[prefix + bone_name]
        heads[j] = bone.head
        tails[j] = bone.tail
        rolls[j] = bone.roll

    bpy.ops.object.mode_set(mode=prev_mode)

    return heads, tails, rolls


def build_skeleton(
    fbx_skeleton, fbx_bone_names, fbx_full_bone_names, fbx_heads, fbx_tails, fbx_rolls
):
    armature, meshes = get_armature_and_meshes()
    for a in bpy.data.actions:
        bpy.data.actions.remove(a)

    for mesh in meshes:
        mesh.vertex_groups.clear()

    deselect_all()

    prev_mode = armature.mode
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="EDIT")

    for bone in armature.data.edit_bones:
        armature.data.edit_bones.remove(bone)

    for parent_name, parent_data in fbx_skeleton.items():
        if parent_name is None or parent_name not in fbx_bone_names:
            continue

        parent_bone = None
        grandparent_name = parent_data["parent"]
        if grandparent_name:
            relatives = [
                child
                for child in fbx_skeleton[grandparent_name]["children"]
                if child in fbx_bone_names
            ]
            mid_relative = get_mid_child(
                grandparent_name, relatives, fbx_full_bone_names, fbx_heads, fbx_tails
            )
            parent_bone = armature.data.edit_bones.get(grandparent_name)
            if (len(relatives) > 1) and (parent_name != mid_relative):
                parent_bone = armature.data.edit_bones.get(
                    grandparent_name + JOINT_CONNECTOR + parent_name
                )

        children = [
            child for child in parent_data["children"] if child in fbx_bone_names
        ]
        mid_child = get_mid_child(
            parent_name, children, fbx_full_bone_names, fbx_heads, fbx_tails
        )
        child_head = fbx_heads[fbx_full_bone_names[parent_name]]

        if len(children) == 0:
            mid_child = get_mid_child(
                parent_name,
                parent_data["children"],
                fbx_full_bone_names,
                fbx_heads,
                fbx_tails,
            )
            child_bone = armature.data.edit_bones.new(parent_name)
            child_bone.head = child_head
            if not mid_child:
                child_bone.tail = fbx_tails[fbx_full_bone_names[parent_name]]
            else:
                child_bone.tail = fbx_heads[fbx_full_bone_names[mid_child]]
            child_bone.roll = fbx_rolls[fbx_full_bone_names[parent_name]]
            child_bone.parent = parent_bone

            leaf_bone = armature.data.edit_bones.new(parent_name + "_VIRTUAL_END")
            leaf_bone.parent = child_bone
            leaf_bone.head = child_bone.tail
            leaf_bone.tail = np.array(child_bone.tail) + np.array(child_bone.vector)

        for child in children:
            if (len(children) > 1) and (child != mid_child):
                child_bone = armature.data.edit_bones.new(
                    parent_name + JOINT_CONNECTOR + child
                )
            else:
                child_bone = armature.data.edit_bones.new(parent_name)
            child_bone.head = child_head
            child_bone.tail = fbx_heads[fbx_full_bone_names[child]]
            child_bone.roll = fbx_rolls[fbx_full_bone_names[parent_name]]
            if parent_bone:
                child_bone.parent = parent_bone

    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.pose.armature_apply()

    bpy.ops.object.mode_set(mode=prev_mode)


def extract_skeleton(prefix):
    armature, _ = get_armature_and_meshes()

    skeleton = {}
    for bone in armature.data.bones:
        bone_name = bone.name[len(prefix) :]
        skeleton[bone_name] = {
            "parent": bone.parent.name[len(prefix) :] if bone.parent else None,
            "children": [child.name[len(prefix) :] for child in bone.children],
        }
    skeleton[None] = {
        "children": [k for k, v in skeleton.items() if v["parent"] is None]
    }
    return skeleton


def export_gltf(output_path, format="GLB"):
    armature, meshes = get_armature_and_meshes()
    deselect_all()
    armature.select_set(True)
    for mesh in meshes:
        mesh.select_set(True)

    bpy.ops.export_scene.gltf(
        filepath=str(output_path),
        export_format=format,
        check_existing=False,
        use_selection=True,
        export_cameras=False,
        export_animations=True,
        export_nla_strips=False,
        export_force_sampling=False,
        export_lights=False,
    )

    return output_path.with_suffix(".glb" if format == "GLB" else ".gltf")


def extract_weight(bone_names, skeleton, prefix):
    _, meshes = get_armature_and_meshes()
    weight = []
    for mesh in meshes:
        for vertex in mesh.data.vertices.values():
            w = np.zeros(len(bone_names))
            for group in vertex.groups:
                bone_name = mesh.vertex_groups[group.group].name[len(prefix) :]
                while bone_name not in bone_names:
                    bone_name = skeleton[bone_name]["parent"]
                w[bone_names[bone_name]] += group.weight
            weight.append(w)
    weight = np.asarray(weight)

    return weight


def adapt_weight(src_bone_names, src_skeleton, src_weight, dst_bone_names):
    dst_weight = np.zeros((src_weight.shape[0], len(dst_bone_names)))
    for i in range(src_weight.shape[0]):
        for src_bone_name, src_bone_index in src_bone_names.items():
            bone_name = src_bone_name
            while bone_name not in dst_bone_names:
                bone_name = src_skeleton[bone_name]["parent"]
            dst_weight[i, dst_bone_names[bone_name]] += src_weight[i, src_bone_index]
    return dst_weight


def assign_weight(bone_names, weight):
    _, meshes = get_armature_and_meshes()
    offset = 0
    for mesh in meshes:
        for bone_name, j in bone_names.items():
            vertex_group = mesh.vertex_groups.new(name=bone_name)

            for i, _ in mesh.data.vertices.items():
                w = weight[offset + i, j]
                vertex_group.add([i], w, "REPLACE")

        offset += len(mesh.data.vertices)


def export_bvh(bvh_path, frame_end):
    bvh_path = bvh_path.with_suffix(".bvh")
    bpy.ops.export_anim.bvh(
        filepath=str(bvh_path),
        check_existing=False,
        frame_end=frame_end,
        rotate_mode="XYZ",
        root_transform_only=True,
    )
    return bvh_path


def adapt_animation(
    fbx_skeleton,
    fbx_bone_names,
    fbx_full_bone_names,
    fbx_heads,
    fbx_tails,
    fbx_matrices,
    bone_names,
):
    armature, _ = get_armature_and_meshes()

    bone_count = len(bone_names)
    frame_count = len(fbx_matrices)
    translations = np.zeros((frame_count, bone_count, 3))
    rotations = np.zeros((frame_count, bone_count, 4))

    for i in range(frame_count):
        for parent_name, parent_data in fbx_skeleton.items():
            if parent_name is None or parent_name not in bone_names:
                continue

            children = [
                child for child in parent_data["children"] if child in fbx_bone_names
            ]
            mid_child = get_mid_child(
                parent_name, children, fbx_full_bone_names, fbx_heads, fbx_tails
            )

            mid_child_bone = armature.pose.bones[parent_name]
            mid_child_idx = bone_names[mid_child_bone.name]

            mid_child_bone.matrix = fbx_matrices[i][fbx_full_bone_names[parent_name]]
            bpy.context.view_layer.update()
            if not parent_data["parent"]:
                mid_child_bone.keyframe_insert(data_path="location", frame=i, index=-1)
                translations[i, mid_child_idx] = (
                    mid_child_bone.bone.matrix_local @ mid_child_bone.location
                )
            else:
                mid_child_bone.location = mathutils.Vector((0, 0, 0))
            mid_child_bone.keyframe_insert(
                data_path="rotation_quaternion", frame=i, index=-1
            )

            arm_rot = local_to_armature_rot(
                mid_child_bone.bone, mid_child_bone.rotation_quaternion.to_matrix()
            )

            rotations[i, mid_child_idx] = to_parent_mat(mid_child_bone).to_quaternion()

            if len(children) > 1:
                for child in children:
                    if child == mid_child:
                        continue
                    child_bone = armature.pose.bones[
                        parent_name + JOINT_CONNECTOR + child
                    ]
                    child_idx = bone_names[child_bone.name]
                    child_bone.rotation_quaternion = armature_to_local_rot(
                        child_bone.bone, arm_rot
                    ).to_quaternion()
                    if not parent_data["parent"]:
                        child_bone.location = armature_to_local_loc(
                            child_bone.bone,
                            mathutils.Matrix.Translation(
                                mid_child_bone.head - child_bone.bone.head_local
                            ),
                        )
                        child_bone.keyframe_insert(
                            data_path="location", frame=i, index=-1
                        )
                        translations[i, child_idx] = translations[i, mid_child_idx]
                    child_bone.keyframe_insert(
                        data_path="rotation_quaternion", frame=i, index=-1
                    )
                    rotations[i, child_idx] = to_parent_mat(child_bone).to_quaternion()

    return translations, rotations


def to_parent_mat(pose_bone):
    pose_mat = pose_bone.matrix
    rest_arm_inv = pose_bone.bone.matrix_local.inverted()
    if pose_bone.parent:
        mat_final = (
            pose_bone.parent.bone.matrix_local
            @ pose_bone.parent.matrix.inverted()
            @ pose_mat
            @ rest_arm_inv
        )
    else:
        mat_final = pose_mat @ rest_arm_inv
    return mat_final


def local_to_armature_rot(bone, matrix):
    rest_matrix = bone.matrix_local.to_3x3()
    rest_matrix_inv = rest_matrix.inverted()
    rest_matrix.resize_4x4()
    rest_matrix_inv.resize_4x4()
    return rest_matrix_inv @ matrix.to_4x4() @ rest_matrix


def armature_to_local_rot(bone, matrix):
    rest_matrix = bone.matrix_local.to_3x3()
    rest_matrix_inv = rest_matrix.inverted()
    rest_matrix.resize_4x4()
    rest_matrix_inv.resize_4x4()
    return rest_matrix @ matrix.to_4x4() @ rest_matrix_inv


def armature_to_local_loc(bone, loc):
    rest_matrix_inv = bone.matrix_local.to_3x3().inverted().to_4x4()
    return (rest_matrix_inv @ loc).to_translation()


def load_gltf(gltf_path):
    delete_all()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bpy.ops.import_scene.gltf(
            filepath=str(gltf_path),
            import_pack_images=False,
            bone_heuristic="BLENDER",
            guess_original_bind_pose=False,
        )


def extract_bone_info(bone_names, prefix):
    armature, _ = get_armature_and_meshes()
    bone_count = len(bone_names)
    parents = np.empty(bone_count, dtype=int)
    positions = np.zeros((bone_count * 2, 3))
    for bone_name, i in bone_names.items():
        bone = armature.data.bones[prefix + bone_name]
        if bone.parent:
            parents[i] = bone_names[bone.parent.name[len(prefix) :]]
        else:
            parents[i] = -1
        index = 2 * i
        positions[index] = bone.head_local
        positions[index + 1] = bone.tail_local
    edges = np.arange(2 * bone_count).reshape((-1, 2))
    return parents, positions, edges


def main():
    parser = argparse.ArgumentParser(
        description="Extracting gltf and skinning weight from fbx"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        help="fbx input root",
    )
    args = parser.parse_args()

    input_root = args.input.resolve(True)
    fbx_paths = list(
        [input_root.with_suffix(".fbx")]
        if input_root.with_suffix(".fbx").exists()
        else input_root.glob("*.fbx")
    )
    for fbx_path in fbx_paths:
        output_dir = fbx_path.with_suffix("")
        output_dir.mkdir(parents=True, exist_ok=True)

        frame_count, prefix = load_fbx(fbx_path)
        fbx_bone_names, fbx_full_bone_names = extract_bone_names(prefix, True)
        fbx_skeleton = extract_skeleton(prefix)
        fbx_matrices = extract_pose_matrices(frame_count, fbx_full_bone_names, prefix)
        fbx_heads, fbx_tails, fbx_rolls = extract_rest_position(
            fbx_full_bone_names, prefix
        )
        fbx_weights = extract_weight(fbx_bone_names, fbx_skeleton, prefix)

        build_skeleton(
            fbx_skeleton,
            fbx_bone_names,
            fbx_full_bone_names,
            fbx_heads,
            fbx_tails,
            fbx_rolls,
        )

        new_bone_names, _ = extract_bone_names("")
        new_skeleton = extract_skeleton("")
        new_weight = adapt_weight(
            fbx_bone_names, fbx_skeleton, fbx_weights, new_bone_names
        )
        assign_weight(new_bone_names, new_weight)

        translations, rotations = adapt_animation(
            fbx_skeleton,
            fbx_bone_names,
            fbx_full_bone_names,
            fbx_heads,
            fbx_tails,
            fbx_matrices,
            new_bone_names,
        )

        gltf_path = export_gltf(output_dir / fbx_path.stem)
        load_gltf(gltf_path)
        weight = extract_weight(new_bone_names, new_skeleton, "")
        export_bvh(output_dir / fbx_path.stem, frame_count - 1)
        parents, positions, edges = extract_bone_info(new_bone_names, "")

        np.savez(
            output_dir / fbx_path.stem,
            weight=weight,
            rotations=rotations,
            translations=translations * FBX_OUTPUT_SCALE,
            parents=parents,
            positions=positions * FBX_OUTPUT_SCALE,
            bone_names=np.array(list(new_bone_names.keys()), dtype=np.str_),
            edges=edges,
        )


if __name__ == "__main__":
    main()
