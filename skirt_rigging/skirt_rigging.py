import bpy
import math
import bmesh
import numpy as np
import copy
import re
from mathutils import Vector
from collections import defaultdict


# Vector angle helpers
def compute_sin(v1, v2):
    v1 = np.array(v1, dtype=np.float64)
    v2 = np.array(v2, dtype=np.float64)
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    crs = np.cross(v2, v1)
    return np.linalg.norm(crs) * np.sign(crs[2])

def compute_cos(v1, v2):
    v1 = np.array(v1, dtype=np.float64)
    v2 = np.array(v2, dtype=np.float64)
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    return np.dot(v1, v2)

def angle(v1, v2):
    sin_t = compute_sin(v1, v2)
    cos_t = compute_cos(v1, v2)
    return np.degrees(np.arctan2(sin_t, cos_t))

def angle_in_xy_plane(v1, v2=[0, -1, 0]):
    v1 = np.array([v1[0], v1[1], 0.0], dtype=np.float64)
    v2 = np.array([v2[0], v2[1], 0.0], dtype=np.float64)
    return angle(v1, v2)

# Detect loop count N (horizontal), M (vertical) using edge loops
def calculate_N_M(obj):
    # Switch to edit mode
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.context.tool_settings.mesh_select_mode = (False, True, False)
    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    
    # select arbitrary
    e = bm.edges[0]
    
    # Find one horizontal edge loop to count N    
    bpy.ops.mesh.select_all(action='DESELECT')
    e.select = True
    bpy.ops.mesh.loop_multi_select(ring=False)
    N = len([e for e in bm.edges if e.select])
    
    # Find one vertical edge loop to count M
    bpy.ops.mesh.select_all(action='DESELECT')
    e.select = True
    bpy.ops.mesh.loop_multi_select(ring=True)
    M = len([e for e in bm.edges if e.select])
    
    # reset mode
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.context.tool_settings.mesh_select_mode = (True, False, False)
    bpy.ops.object.mode_set(mode='OBJECT')
    return N, M

# Create vertex groups with naming: points.XXX.YYY
def allocate_vertex_group(skirt):    
    N, M  = calculate_N_M(skirt)
    print(f"Detected: N (horizontal) = {N}, M (vertical) = {M}")

    bpy.context.view_layer.objects.active = skirt
    bpy.ops.object.mode_set(mode='OBJECT')

    # create bmesh 
    bm = bmesh.new()
    bm.from_mesh(skirt.data)
    bm.verts.ensure_lookup_table()

    # Find initial vertices near -Y axis
    start_candidates = [v for v in bm.verts if abs(angle_in_xy_plane(v.co)) <= 5 and v.co.y < 0]
    print(len(start_candidates))
    if not start_candidates:
        raise ValueError("No vertex found near -Y axis")

    sorted_init_vtx = sorted(start_candidates, key=lambda v: -v.co.z)
    
    # Remove existing vertex groups
    for vg in skirt.vertex_groups:
        skirt.vertex_groups.remove(vg)

    vg_candidates = {}


    # Traverse vertices and assign to groups
    for j, v_init in enumerate(sorted_init_vtx):
        v_cur = v_init
        i = 0
        vg_candidates[v_init.index] = f"points.{str(i).zfill(3)}.{str(j).zfill(3)}"
        print(f"points.{str(i).zfill(3)}.{str(j).zfill(3)}")
        
        i = 1
        while(i < N):
            connected_vtx = [e.other_vert(v_cur) for e in v_cur.link_edges]
            found = False
            
            for v_next in connected_vtx:
                # pass visited 
                if v_next.index in vg_candidates.keys():
                    continue
                
                # pass horizontal clockwise neighbor
                if angle_in_xy_plane(v_cur.co, v_next.co) >= 0.0:
                    continue
                
                # pass vertical neighbor
                if np.abs(compute_cos(v_cur.co - v_next.co, [0, 0, 1])) > 0.5:
                    continue

                # allocate vtx
                vg_candidates[v_next.index] = f"points.{str(i).zfill(3)}.{str(j).zfill(3)}"
                print(f"points.{str(i).zfill(3)}.{str(j).zfill(3)}")

                # set
                v_cur = v_next
                i += 1
                found = True
                print(f"{i}, {j}")
                break

            if not found:
                print(f"finish loop")
                break
        
    bm.to_mesh(skirt.data)
    bm.free()    

    # add vg group
    for vg_index in vg_candidates.keys():
        vg = skirt.vertex_groups.new(name=vg_candidates[vg_index])
        vg.add([vg_index], 1.0, 'REPLACE')
  
    # add pin group          
    pin_group_vg = skirt.vertex_groups.new(name='pin_group')
    vg_pattern = re.compile(r"points\.(\d+)\.000")
    for vg_index in vg_candidates.keys():
        vg_name = vg_candidates[vg_index]
        match = vg_pattern.match(vg_name)
        if match:
            pin_group_vg.add([vg_index], 1.0, 'REPLACE')  

# add clothe physics
def add_cloth_physics(skirt):
    # Add cloth physics
    if skirt.modifiers.get("Cloth") is None:
        cloth_mod = skirt.modifiers.new(name="Cloth", type='CLOTH')
    else:
        cloth_mod = skirt.modifiers["Cloth"]

    # Set cloth physics 
    cloth_settings = cloth_mod.settings

    # Set Pin Group
    cloth_settings.vertex_group_mass = "pin_group"

    # 적용 후 3D 뷰 업데이트
    bpy.context.view_layer.update()

# Assign bones to bone collections based on class names in bone names        
def assign_bones_to_collections(armature_obj, class_list):
    arm = armature_obj.data

    collection_map = {}
    for class_name in class_list:
        coll = arm.collections.get(class_name)
        if not coll:
            coll = arm.collections.new(class_name)
        collection_map[class_name] = coll

    for bone in arm.bones:
        for class_name in class_list:
            if class_name in bone.name:                
                for coll in bone.collections:
                    coll.unassign(bone)                
                collection_map[class_name].assign(bone)
                break

# Create armature and build physics/FK bones            
def create_aramature(skirt, rig_name="skirt_rig"):
    vg_pattern = re.compile(r"points\.(\d+)\.(\d+)")
    vg_map = defaultdict(dict)

    for vg in skirt.vertex_groups:
        match = vg_pattern.match(vg.name)
        if match:
            i, j = int(match.group(1)), int(match.group(2))
            vg_map[i][j] = vg.index
        
    vertex_group_to_index = {vg.index: vg for vg in skirt.vertex_groups}

    armature = bpy.data.armatures.new(rig_name)
    armature_obj = bpy.data.objects.new(rig_name, armature)
    bpy.context.collection.objects.link(armature_obj)
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='EDIT')
    bones = armature.edit_bones

    # Add root bone
    bone_name = f"root"
    root_bone = bones.new(bone_name)
    root_bone.head = Vector((0, 0, 0))
    root_bone.tail = Vector((0, 0, 1))
            
    for i in sorted(vg_map.keys()):
        column = vg_map[i]
        sorted_j = sorted(column.keys())

        for y in range(1, len(sorted_j)):
            j_prev = sorted_j[y - 1]
            j_cur = sorted_j[y]

            vg_index_prev = column[j_prev]
            vg_index_cur = column[j_cur]

            verts_prev = [v for v in skirt.data.vertices if vg_index_prev in [vg.group for vg in v.groups]]
            verts_cur = [v for v in skirt.data.vertices if vg_index_cur in [vg.group for vg in v.groups]]
            if not verts_prev or not verts_cur:
                continue

            head = verts_prev[0].co
            tail = verts_cur[0].co

            # Physics bone
            bone_name = f"skirt_physics.{i:03d}.{j_cur:03d}"
            bone = bones.new(bone_name)
            bone.head = head
            bone.tail = tail
            
            if y > 1:
                parent_name = f"skirt_physics.{i:03d}.{sorted_j[y - 1]:03d}"
                if parent_name in bones:
                    bone.parent = bones[parent_name]
            elif y == 1:
                bone.parent = root_bone

            # FK bone
            bone_name = f"skirt_FK.{i:03d}.{j_cur:03d}"
            bone = bones.new(bone_name)
            bone.head = head
            bone.tail = tail
            
            if y > 1:
                parent_name = f"skirt_FK.{i:03d}.{sorted_j[y - 1]:03d}"
                if parent_name in bones:
                    bone.parent = bones[parent_name]   
            elif y == 1:
                bone.parent = root_bone
                    
    bpy.ops.object.mode_set(mode='OBJECT')
    
    assign_bones_to_collections(armature_obj, ["FK", "physics", "root"])
    
# Add Damped Track constraint to physics bones
def add_physics_bone_damped_track_constraint(skirt, armature_obj):
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='POSE')

    bone_pattern = re.compile(r"skirt_physics\.(\d+)\.(\d+)")

    for bone in armature_obj.pose.bones:
        match = bone_pattern.match(bone.name)
        if not match:
            continue

        x, y = match.groups()
        subtarget_name = f"points.{x}.{y}"

        # Constraint 추가
        constraint = bone.constraints.new(type='DAMPED_TRACK')
        constraint.name = "Damped Track"
        constraint.target = skirt
        constraint.subtarget = subtarget_name
        constraint.head_tail = 1.0  # 보통 Tail이 향하게 하려면 1.0

    bpy.ops.object.mode_set(mode='OBJECT')

# Add FK bone Copy Rotation constraint to follow physics bones
def add_FK_bone_copy_constraint(skirt, armature_obj, rig_name="skirt_rig"):
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='POSE')  
    
    bone_pattern = re.compile(r"skirt_FK\.(\d+)\.(\d+)")

    for bone in armature_obj.pose.bones:
        match = bone_pattern.match(bone.name)
        if not match:
            continue
        
        cons = bone.constraints.new('COPY_ROTATION')
        cons.name = "CTRL:" + cons.name + "_physics"
        
        cur_rig_name = ".".join(bone.name.split(".")[:-1])
        cons.target = bpy.data.objects[rig_name]
        cons.subtarget = bone.name.replace("FK", "physics") 
        cons.mix_mode = 'BEFORE'
        cons.target_space = 'LOCAL'
        cons.owner_space = 'LOCAL'


# Get skirt object
skirt = bpy.data.objects.get("skirt")
if not skirt:
    raise ValueError("'skirt' object not found")

# Execute rigging process
allocate_vertex_group(skirt)
add_cloth_physics(skirt)

create_aramature(skirt)

armature_obj = bpy.data.objects.get("skirt_rig")

add_physics_bone_damped_track_constraint(skirt, armature_obj)
add_FK_bone_copy_constraint(skirt, armature_obj)
