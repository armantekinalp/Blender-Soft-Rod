import bpy
import numpy as np
import colorsys

#treating these as globals

npz_file_path = "/Users/rohitharish/Downloads/br2_data.npz"

npz_data = np.load(npz_file_path)

file_position = npz_data['position_rod']

rod_frames = []

for i in range (len(file_position)):
  curr_frame = []
  for j in range (len(file_position[i][0])):
      x = file_position[i][0][j]
      y = file_position[i][1][j]
      z = file_position[i][2][j]
      curr_frame.append(np.array([x,y,z]))
  rod_frames.append(curr_frame)
rod_frames = np.array(rod_frames)

class Sphere:
    def __init__(self, location, radius):
        self.obj = self.create_sphere(location, radius)
       
    def create_sphere(self, location, radius):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=location)
        return bpy.context.active_object
   
    def update_position(self, location):
        self.obj.location.z  = location[2]
        self.obj.location.y = location[1]
        self.obj.location.x = location[0]

class Frustum:
    def __init__(self, pos1, pos2, radius1, radius2):
        self.obj = self.create_frustum(pos1, pos2, radius1, radius2)
        self.mat = bpy.data.materials.new(name="cyl_mat")
        self.obj.active_material = self.mat

    def create_frustum(self, pos1, pos2, radius1, radius2):
        depth, center, angles = self.calc_frust_orientation(pos1, pos2)
        bpy.ops.mesh.primitive_cone_add(radius1 = radius1, radius2 = radius2, depth=1, location=center)
        frustum = bpy.context.active_object
        frustum.rotation_euler = (0, angles[1], angles[0])
        frustum.scale[2] = depth
        return frustum

    def calc_frust_orientation(self, pos1, pos2):
        pos1 = np.array(pos1)
        pos2 = np.array(pos2)
        depth = np.linalg.norm(pos2 - pos1)
        dz =  pos2[2] - pos1[2]
        dy = pos2[1] - pos1[1]
        dx = pos2[0] - pos1[0]
        center = (pos1 + pos2) / 2
        phi = np.arctan2(dy, dx)
        theta = np.arccos(dz/depth)
        angles = np.array([phi, theta])
        return depth, center, angles
   
    def update_position(self, pos1, pos2):
        depth, center, angles = self.calc_frust_orientation(pos1,pos2)
        self.obj.location = (center[0],center[1],center[2])
        self.obj.rotation_euler = (0, angles[1], angles[0])
        self.obj.scale[2] = depth
       
        #computing deformation heat-map
        max_def = 0.07
       
        h = -np.sqrt(self.obj.location[0]**2 + self.obj.location[2]**2)/max_def + 240/360
        v = np.sqrt(self.obj.location[0]**2 + self.obj.location[2]**2)/max_def*0.5 + 0.5

        r, g, b = colorsys.hsv_to_rgb(h, 1, v)
        self.update_color(r,g,b,1)
       
   
    def update_color(self, r,g,b,a):
        self.mat.diffuse_color = (r,g,b,a)

# Clear existing mesh objects in the scene
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_by_type(type='MESH')
bpy.ops.object.delete()
for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)




#allow different radii
radii = []
start = 0.005

for i in range (len(rod_frames[0])):
    num = i/10000
    radii.append(start + num)
#creates spheres

spheres = []
cnt = 0
for point in rod_frames[0]:
    spheres.append(Sphere(point, radius = radii[cnt]))
    cnt = cnt + 1
#create cylinders
frustums = []
for i in range(len(spheres)-1):
    frustums.append(Frustum(spheres[i].obj.location, spheres[i+1].obj.location, radii[i+1], radii[i]))


#for each time step, update all sphere and cylinder positions and write object to keyframe
for time_step in range(len(rod_frames)):
    for s in range (len(spheres)):
        spheres[s].update_position(rod_frames[time_step][s])
        #adding to keyframe
        spheres[s].obj.keyframe_insert(data_path="location", frame=time_step)
    for c in range (len(frustums)):
        frustums[c].update_position(spheres[c].obj.location, spheres[c+1].obj.location)
        # adding to keyframe
        frustums[c].obj.keyframe_insert(data_path="location", frame=time_step)
        frustums[c].obj.keyframe_insert(data_path="rotation_euler", frame=time_step)
        frustums[c].obj.keyframe_insert(data_path="scale", frame=time_step)
        frustums[c].mat.keyframe_insert(data_path="diffuse_color", frame=time_step)