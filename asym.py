from two_box import TwoBoxFamily, TwoBox, show_solid, two_box_faces
import numpy as np 
import pandas as pd
from scipy.spatial.transform import Rotation
from copy import deepcopy

def to_homogeneous(pts: np.ndarray):
	"""
	Given an ndarray pts of any size, adds a column of ones to the last dimension.
	So if pts has size (3, ) output is size (4, ), if size (N, 2), output is (N,3), 
	If input is (B, N, 2), output is (B,N,3), and so on...
	"""
	ONES = np.ones( tuple(i if j < pts.ndim-1 else 1 for j, i in enumerate(pts.shape)) )
	return np.concatenate((pts, ONES), axis=-1)


def inner_asym1(xyz, amount_of_rotation):
	faces = two_box_faces()
	u,s,vt = np.linalg.svd(to_homogeneous(xyz[faces]))
	planes = vt[:, -1]
	planes /= np.linalg.norm(planes[:, :3], axis=1, keepdims=True)
	planes = pd.DataFrame(planes, index=faces.index, columns = ['nx', 'ny', 'nz', 'd'])
	# Generate Axes of rotation for the sides
	side_faces = ["big_right", 'big_left','small_right', 'small_left']
	index_of_side_faces = [i for i,f in enumerate(faces.index) if f in side_faces]
	side_normals = planes.loc[side_faces, ['nx', 'ny', 'nz']]
	u,s,vt = np.linalg.svd(side_normals.values[:, None,:])
	basis = vt[:, :, 1:]
	theta = rng.uniform(0,2*np.pi, size=len(basis))
	axes_of_rotation = np.cos(theta)[:, None] * basis[:,:, 0] + \
							np.sin(theta)[:, None] * basis[:, :, 1]
	# amount_of_rotation = np.radians(np.array([20,20,20,20]))
	axes = axes_of_rotation * amount_of_rotation[:, None]
	r = Rotation.from_rotvec(axes)
	rotated_side_normals = np.squeeze(r.as_matrix() @ side_normals.values[:, :, None])
	# Edit Side Planes 
	centers = np.mean(xyz[faces], axis=1)
	intercepts = -np.vecdot(rotated_side_normals, centers[index_of_side_faces])
	rotated_side_planes = np.hstack((rotated_side_normals, intercepts[:, None]))
	new_planes = planes.copy()
	new_planes.loc[side_faces] = rotated_side_planes
	faces_per_point = [
		['big_bottom', 'big_back', 'big_right'], 
		['big_bottom', 'big_back', 'big_left'],
		['big_bottom', 'big_front', 'big_right'],
		['big_bottom', 'big_front', 'big_left'],
		['big_top', 'big_back', 'big_right'], 
		['big_top', 'big_back', 'big_left'],
		['big_top', 'big_front', 'big_right'],
		['big_top', 'big_front', 'big_left'],
		['small_bottom', 'small_back', 'small_right'], 
		['small_bottom', 'small_back', 'small_left'],
		['small_bottom', 'small_front', 'small_right'],
		['small_bottom', 'small_front', 'small_left'],
		['small_top', 'small_back', 'small_right'], 
		['small_top', 'small_back', 'small_left'],
		['small_top', 'small_front', 'small_right'],
		['small_top', 'small_front', 'small_left']
		]
	t_dct = {name:i for i, name in enumerate(faces.index)}
	faces_per_point_index = [[t_dct[r] for r in row] for row in faces_per_point]
	planes_faces = new_planes.values[faces_per_point_index]
	normals_faces = planes_faces[:, :, :3]
	intercepts_faces = planes_faces[:, :, 3:]
	new_xyz = np.squeeze(np.linalg.solve(normals_faces, -intercepts_faces))
	return new_xyz


def passes_checks_asym1(xyz):
	c1 = (xyz[2,0] > xyz[8,0]) and (xyz[8,0] > xyz[9,0]) and (xyz[9,0] > xyz[3,0])
	c2 = (xyz[10,0] > xyz[11,0]) and (xyz[14,0] > xyz[15,0]) and (xyz[12,0] > xyz[13,0])
	c3 = (xyz[0,0] > xyz[1,0]) and (xyz[4,0] > xyz[5,0]) and (xyz[6,0] > xyz[7,0])
	c4 = -xyz[6,1]*xyz[2,0] + xyz[12,1]*xyz[2,0] - xyz[12,1]*xyz[6,0] + xyz[12,0]*xyz[6,1] <= 0
	c5 = -xyz[7,1]*xyz[3,0] + xyz[13,1]*xyz[3,0] - xyz[13,1]*xyz[7,0] + xyz[13,0]*xyz[7,1] >= 0
	return c1 & c2 & c3 &c4&c5

def asym1(obj, amount_of_rotation, max_starts=50):
	new_obj = deepcopy(obj)
	for i in range(max_starts):
		new_xyz = inner_asym1(new_obj.xyz, amount_of_rotation)
		if passes_checks_asym1(new_xyz):
			new_obj.xyz = new_xyz
			return new_obj
	return None




if __name__ == "__main__":
	rng = np.random.default_rng()
	angle_dct = {
		"big_right":rng.uniform(10,20) * rng.choice([1,-1]), 
		"small_right":rng.uniform(10,15) * rng.choice([1,-1]), 
		"big_top":rng.uniform(-20,20), 
		"big_front":rng.uniform(-20,20), 
		"big_back":rng.uniform(-20,20),
		"small_top":rng.uniform(-10,10),
		"small_front":rng.uniform(-10,10),
		"tilt_big_right":rng.uniform(0,360),
		"tilt_small_right":rng.uniform(0,360)
		}


	tbf = TwoBoxFamily(width_x=7, height_y=7, depth_z=7, angles=angle_dct, rng=rng)
	obj = tbf.sample()
	asym_obj = asym1(obj, np.radians(np.array([20,20,20,20])))
	show_solid(asym_obj.xyz, asym_obj.triangles, edges=asym_obj.edges)


