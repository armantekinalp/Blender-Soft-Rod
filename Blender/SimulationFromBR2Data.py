import bpy
import numpy as np
import colorsys

#treating these as globals

npz_file_path = "C:/Users/darren/Downloads/br2_data.npz"
npz_data = np.load(npz_file_path)

file_position = npz_data['position_rod']

double_listx = []
double_listy = []
double_listz = []

#remember to initialize to zeros at first

for i in range (26):
  file_xpos = []
  file_ypos = []
  file_zpos = []
  for j in range (40):
      file_xpos.append(file_position[i][0][j])
      file_ypos.append(file_position[i][1][j])
      file_zpos.append(file_position[i][2][j])
  double_listx.append(file_xpos)
  double_listy.append(file_ypos)
  double_listz.append(file_zpos)


class Sphere:
    def __init__(self, location):
        self.obj = self.create_sphere(location)
        
    def create_sphere(self, location):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.0025, location=location)
        return bpy.context.active_object
    
    def update_position(self, newx, newy, newz):
        self.obj.location.z  = newz
        self.obj.location.y = newy
        self.obj.location.x = newx

class Cylinder:
    def __init__(self, pos1, pos2):
        self.obj = self.create_cylinder(pos1, pos2)
        self.mat = bpy.data.materials.new(name="cyl_mat")
        self.obj.active_material = self.mat

    def create_cylinder(self, pos1, pos2):
        depth, center, angles = self.calc_cyl_orientation(pos1, pos2)
        bpy.ops.mesh.primitive_cylinder_add(radius=0.005, depth=1, location=center)
        cylinder = bpy.context.active_object
        cylinder.rotation_euler = (0, angles[1], angles[0])
        cylinder.scale[2] = depth
        return cylinder

    def calc_cyl_orientation(self, pos1, pos2):
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
        depth, center, angles = self.calc_cyl_orientation(pos1,pos2)
        self.obj.location = (center[0],center[1],center[2])
        self.obj.rotation_euler = (0, angles[1], angles[0])
        self.obj.scale[2] = depth
        
        #computing deformation heat-map
        max_def = 0.07
        
        h = -np.sqrt(self.obj.location[0]**2 + self.obj.location[2]**2)/max_def + 240/360
        r, g, b = colorsys.hsv_to_rgb(h, 1, 1)
        self.update_color(r,g,b,1)
        
    
    def update_color(self, r,g,b,a):
        self.mat.diffuse_color = (r,g,b,a)

# Clear existing mesh objects in the scene
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_by_type(type='MESH')
bpy.ops.object.delete()
for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)



#creates spheres
spheres = []

for i in range(len(double_listx[0])):
    spheres.append(Sphere((double_listx[0][i],double_listy[0][i],double_listz[0][i])))
#create cylinders
cylinders = []
for i in range(len(spheres)-1):
    cylinders.append(Cylinder(spheres[i].obj.location, spheres[i+1].obj.location))


#for each time step, update all sphere and cylinder positions and write object to keyframe
for time_step in range(len(double_listx)):
    for s in range (len(spheres)):
        spheres[s].update_position(double_listx[time_step][s], double_listy[time_step][s], double_listz[time_step][s])
        #adding to keyframe
        spheres[s].obj.keyframe_insert(data_path="location", frame=time_step)
    for c in range (len(cylinders)):
        cylinders[c].update_position(spheres[c].obj.location, spheres[c+1].obj.location)
        # adding to keyframe
        cylinders[c].obj.keyframe_insert(data_path="location", frame=time_step)
        cylinders[c].obj.keyframe_insert(data_path="rotation_euler", frame=time_step)
        cylinders[c].obj.keyframe_insert(data_path="scale", frame=time_step)
        cylinders[c].mat.keyframe_insert(data_path="diffuse_color", frame=time_step)

