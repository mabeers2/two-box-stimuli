import numpy as np 
from scipy.sparse import csr_array, vstack
import matplotlib.pyplot as plt
import pandas as pd
import polytope as pc
import pyomo.environ as pyo
import pyvista as pv
from scipy.spatial.transform import Rotation


def get_position_dct():
	intercepts = ['big_right', 'big_top', 'big_front', 'big_back', 'small_top', 'small_front', "small_right"]
	right = list(range(0,16,2))
	counter = 0
	dct = {}
	for i in right:
		for letter in list('xyz'):
			dct[f"{letter}{i}"] = counter 
			counter += 1
	for intercept in intercepts:
		dct[intercept] = counter 
		counter += 1
	return dct 


def build_equality_mat_planarity(normals, faces_right, position_dct):
	row_counter = 0
	row = []
	col = []
	data = []
	letters = list("xyz")
	for name in faces_right:
		for i in faces_right[name]:
			for letter, value in zip(letters, normals.loc[name].tolist()):
				row.append(row_counter)
				col.append(position_dct[f"{letter}{i}"])
				data.append(value)
			row.append(row_counter)
			col.append(position_dct[name])
			data.append(1)
			row_counter += 1
	return csr_array((data, (row, col)), shape=(row_counter, len(position_dct))), np.zeros(row_counter)


def build_equality_mat_single_values(names, values, position_dct):
	row_counter = 0
	row = []
	col = []
	data = []
	letters = list("xyz")
	b = []
	for name, value in zip(names, values):
		row.append(row_counter)
		col.append(position_dct[name])
		data.append(1)
		b.append(value)
		row_counter += 1
	return csr_array((data, (row, col)), shape=(row_counter, len(position_dct))), np.array(b)


def _build_matrix_Ab(normals, faces_right, names, values, position_dct):
	A_eq_planes, b_eq_planes = build_equality_mat_planarity(normals, faces_right, position_dct)
	A_eq_equality, b_eq_equality = build_equality_mat_single_values(names, values, position_dct)
	A_eq = vstack((A_eq_planes, A_eq_equality))
	b_eq = np.hstack((b_eq_planes, b_eq_equality))
	return A_eq, b_eq


def build_ub_mat(lower, upper, position_dct):
	row_counter = 0
	row = []
	col = []
	data = []
	b = []
	for l, u in zip(lower, upper):
		if isinstance(l, str) and isinstance(u, str):
			row.append(row_counter)
			col.append(position_dct[l])
			data.append(1.)
			row.append(row_counter)
			col.append(position_dct[u])
			data.append(-1.)
			b.append(0)
		elif isinstance(l, str) and isinstance(u, (int, float)):
			row.append(row_counter)
			col.append(position_dct[l])
			data.append(1.)
			b.append(u)
		elif isinstance(l, (int, float)) and isinstance(u, str):	
			row.append(row_counter)
			col.append(position_dct[u])
			data.append(-1)
			b.append(-l)
		row_counter += 1
	return csr_array((data, (row, col)), shape=(row_counter, len(position_dct))), np.array(b)


def ub_edge_length_mat(j, dx, dy, dz, eps=.1):
	row_counter = 0
	row = []
	col = []
	data = []
	b = []
	# y4 > y0 + eps * dy <==> y0 - y4 <= -eps * dy
	row.append(row_counter)
	col.append(j['y0'])
	data.append(1)
	row.append(row_counter)
	col.append(j['y4'])
	data.append(-1)
	b.append(-eps * dy)
	row_counter += 1
	# y6 > y2 + eps * dy <==> y2 - y6 <= -eps * dy
	row.append(row_counter)
	col.append(j['y2'])
	data.append(1)
	row.append(row_counter)
	col.append(j['y6'])
	data.append(-1)
	b.append(-eps * dy)
	row_counter += 1
	# z0 - z2 >= eps * dz <==> z2-z0 <= -eps * dz
	row.append(row_counter)
	col.append(j['z2'])
	data.append(1)
	row.append(row_counter)
	col.append(j['z0'])
	data.append(-1)
	b.append(-eps * dz)
	row_counter += 1
	# z4 - z6 >= eps * dz <==> z6-z4 <= -eps * dz
	row.append(row_counter)
	col.append(j['z6'])
	data.append(1)
	row.append(row_counter)
	col.append(j['z4'])
	data.append(-1)
	b.append(-eps * dz)
	row_counter += 1
	# z12 - z14 >= eps * dz <==> z14-z12 <= -eps * dz
	row.append(row_counter)
	col.append(j['z14'])
	data.append(1)
	row.append(row_counter)
	col.append(j['z12'])
	data.append(-1)
	b.append(-eps * dz)
	row_counter += 1
	# z8 - z10 >= eps * dz <==> z10-z8 <= -eps * dz
	row.append(row_counter)
	col.append(j['z10'])
	data.append(1)
	row.append(row_counter)
	col.append(j['z8'])
	data.append(-1)
	b.append(-eps * dz)
	row_counter += 1
	# y12 - y8 >= eps * dy <==> y8-y12 <= -eps * dy
	row.append(row_counter)
	col.append(j['y8'])
	data.append(1)
	row.append(row_counter)
	col.append(j['y12'])
	data.append(-1)
	b.append(-eps * dy)
	row_counter += 1
	# y14 - y10 >= eps * dy <==> y8-y12 <= -eps * dy
	row.append(row_counter)
	col.append(j['y10'])
	data.append(1)
	row.append(row_counter)
	col.append(j['y14'])
	data.append(-1)
	b.append(-eps * dy)
	row_counter += 1
	# x2 - x8 >= eps * dx <==> x8-x2 <= -eps * dx
	row.append(row_counter)
	col.append(j['x8'])
	data.append(1)
	row.append(row_counter)
	col.append(j['x2'])
	data.append(-1)
	b.append(-eps * dx)
	row_counter += 1
	return csr_array((data, (row, col)), shape=(row_counter, len(j))), np.array(b)


def make_normals(angle_dct):
	normals = dict()
	normals['big_top'] = np.array([0, np.cos(np.radians(angle_dct['big_top'])), np.sin(np.radians(angle_dct['big_top']))])
	normals['big_front'] = np.array([0, np.sin(np.radians(angle_dct['big_front'])), -np.cos(np.radians(angle_dct['big_front']))])
	normals['big_back'] = np.array([0, np.sin(np.radians(angle_dct['big_back'])), np.cos(np.radians(angle_dct['big_back']))])
	normals['small_top'] = np.array([0, np.cos(np.radians(angle_dct['small_top'])), np.sin(np.radians(angle_dct['small_top']))])
	normals['small_front'] = np.array([0, np.sin(np.radians(angle_dct['small_front'])), -np.cos(np.radians(angle_dct['small_front']))])
	axis = np.array([
		0, 
		np.cos(np.radians(angle_dct['tilt_big_right'])), 
		np.sin(np.radians(angle_dct['tilt_big_right']))
		])
	R = Rotation.from_rotvec(axis * np.radians(angle_dct['big_right']))
	normals['big_right'] = R.apply(np.array([1,0,0]))
	axis = np.array([
		0, 
		np.cos(np.radians(angle_dct['tilt_small_right'])), 
		np.sin(np.radians(angle_dct['tilt_small_right']))
		])
	R = Rotation.from_rotvec(axis * np.radians(angle_dct['small_right']))
	normals['small_right'] = R.apply(np.array([1,0,0]))
	normals['bottom'] = np.array([0,-1,0])
	return pd.DataFrame(normals, index=['nx', 'ny', 'nz']).T


def identify_extreme_vertices(normals):
	"""
	The two box object has specified length, width, height. 
	This function determines, for a set of normals, which point 
	determines that equality 
	"""
	large_i = np.array([0,2,4,6])
	large_i_faces = [
		['bottom', 'big_back', 'big_right'],
		['bottom', 'big_front', 'big_right'],
		['big_top', 'big_back', 'big_right'],
		['big_top', 'big_front', 'big_right']
	]
	A = np.stack([normals.loc[f, :].values for f in large_i_faces])
	b = 10 * np.ones((len(A), 3, 1))
	xyz = np.linalg.solve(A,b).reshape(-1,3)
	xmax, ymax, zmax = large_i[np.argmax(xyz, axis=0)]
	dct = {"x_max":f"x{xmax}", "y_max":f"y{ymax}", "z_max":f"z{zmax}"}
	small_i = [10,14]
	small_i_faces = [
		["bottom", "small_front", "small_right"],
		['small_top', 'small_front', 'small_right']
	]
	A = np.stack([normals.loc[f, :].values for f in small_i_faces])
	b = 10 * np.ones((len(A), 3, 1))
	xyz = np.linalg.solve(A,b).reshape(-1,3)
	zmin = small_i[np.argmin(xyz[:, 2])]
	dct['z_min'] = f"z{zmin}"
	return dct


def line_segment_intersection(s1, s2):
	pts = np.hstack((s1, s2))
	order = np.argsort(pts)
	hypothetical = (pts[order[1]] + pts[order[2]]) / 2
	if not (min(s1) <= hypothetical <= max(s1)):
		return None
	else:
		return pts[order[[1,2]]]


def _build_matrix_Q(j):
	Q = np.zeros((len(j), len(j)))
	Q[j['x12'],j['y6']] = 1
	Q[j['x2'], j['y12']] = 1
	Q[j['x6'], j['y12']] = -1
	Q[j['x2'], j['y6']] = -1
	Q = Q + Q.T
	return Q


def _build_matrix_Cd(lower, upper, j, dx, dy, dz, eps=.1):
	A_ub1, b_ub1 = build_ub_mat(lower, upper, j)
	A_ub2, b_ub2 = ub_edge_length_mat(j, dx, dy, dz, eps=eps)
	A_ub = vstack((A_ub1, A_ub2))
	b_ub = np.hstack((b_ub1, b_ub2))
	return A_ub, b_ub


def check_constraints_possible(n, P, q, r, C, d):
	model = pyo.ConcreteModel()
	model.I = pyo.RangeSet(1, n)
	model.w = pyo.Var(model.I, domain = pyo.Reals)
	model.iC = pyo.RangeSet(1, len(C))

	@model.Objective(sense=pyo.maximize) 
	def objective_rule(model):
		return 0

	@model.Constraint()
	def quadratic_constraint(model):
		return sum(P[i-1,j-1]*model.w[i]*model.w[j] for i in model.I for j in model.I) + sum(q[i-1] * model.w[i] for i in model.I) + r <= 0

	@model.Constraint(model.iC)
	def linear_constraint(model, i):
		return sum(C[i-1, j-1] * model.w[j] for j in model.I) <= d[i-1]

	solver = pyo.SolverFactory('ipopt')
	results = solver.solve(model, tee=False)
	soln = np.array([value for key, value in model.w.extract_values().items()])
	solved = results.solver.termination_condition == pyo.TerminationCondition.optimal
	return {"soln_exists":solved, "solution":soln}


def random_weight_uniform(size, rng):
    w = rng.exponential(size=size)
    return w / w.sum(axis=-1, keepdims=True)


def get_triangles_edges():
	faces = np.array([
	[0,1,3,2], 
	[4,5,7,6], 
	[0,2,6,4], 
	[0,1,5,4], 
	[1,5,7,3], 
	[2,3,7,6]])
	faces = np.vstack((faces, faces + 8))
	edges = set()
	for face in faces:
		for i in range(4):
			low = min(face[i], face[(i+1)%4])
			high = max(face[i], face[(i+1)%4])
			edges.add((low,high))
	edges.remove((2,3))
	edges.add((3,9))
	edges.add((2,8))
	edges = np.array(list(edges))
	triangles = [
		[8,2,0],[1,8,0],[9,8,1], [3,9,1],[9,11,8], [11,10,8],
		[5,1,0],[0,4,5],
		[2,4,0], [2,6,4],
		[7,5,4], [4,6,7],
		[1,5,7], [7,3,1],
		[2,8,12], [2,12,6], [12,7,6], [12,13,7],[7,13,3], [3,13,9],
		[10,11,15],[15,14,10], 
		[15,13,12], [12,14,15], 
		[8,10,14],[14,12,8], 
		[11,9,13], [13,15,11]]
	return np.array(triangles), edges


def hit_and_run(n, x0, P, q, r, C, d, rng, warm_up=10):
	N = warm_up + n
	pts = np.zeros((N, len(x0)))
	pts[0] = x0
	for i in range(1, N):
		s = rng.normal(size=len(x0))
		s /= np.linalg.norm(s)
		alpha = (d - C @ pts[i-1]) / (C @ s)
		alpha_high = np.min(alpha[alpha > 0])
		alpha_low = np.max(alpha[alpha < 0])
		a = s.T @ P @ s
		b = (2 * s @ P @ x0 + q @ s)
		c = x0 @ P @ x0 + q @ x0 + r 
		discriminant = b**2 - 4*a*c
		if discriminant >= 0:
			root1 = (-b + np.sqrt(discriminant)) / (2 * a)
			root2 = (-b - np.sqrt(discriminant)) / (2 * a)
			central_alpha = (root1 + root2) / 2
			if a * central_alpha ** 2 + b*central_alpha + c <= 0:
				# central region desired, alpha must be in intersection of:
				# [alpha_low, alpha_high] & [min(root1, root2), max(root1, root2)]
				# I think geometrically this never happens but just code it up to make sure. 
				intersection = line_segment_intersection([root1, root2], [alpha_low, alpha_high])
				if intersection is not None:
					alpha_star = rng.uniform(*intersection)
				else:
					raise ValueError("Quadratic and Linear Constraints not compatible. Should never happen.")
			else:
				# outer regions desired 
				intersection_low = line_segment_intersection([-np.inf, min(root1, root2)], [alpha_low, alpha_high])
				intersection_high = line_segment_intersection([np.inf, max(root1, root2)], [alpha_low, alpha_high])
				if intersection_low is not None and intersection_high is None:
					alpha_star = rng.uniform(*intersection_low)
				elif intersection_low is None and intersection_high is not None:
					alpha_star = rng.uniform(*intersection_high)
				elif intersection_low is not None and intersection_high is not None:
					raise ValueError("Quadratic and Linear Constraints not compatible. Again, should never happen.")
				else:
					# Intersection low and intersection high both exist. 
					length_low = intersection_low[1] - intersection_low[0] 
					length_high = intersection_high[1] - intersection_high[0] 
					p = length_low / (length_low + length_high)
					if rng.uniform() < p:
						alpha_star = rng.uniform(*intersection_low)
					else:
						alpha_star = rng.uniform(*intersection_high)
		else:
			raise ValueError("Quadratic either always satisfied or never satisfied. If never satisfied, then no solution at all and this should never happen.")
		pts[i] = pts[i-1] + alpha_star * s
		if not np.all(C @ pts[i] <= d):
			print("help",np.where(C @ pts[i] > d)[0])
			break
	return pts[warm_up:]

def two_box_faces():
	faces = np.array([
	[0,1,3,2], 
	[4,5,7,6], 
	[0,2,6,4], 
	[0,1,5,4], 
	[1,5,7,3], 
	[2,3,7,6]])
	faces = np.vstack((faces, faces + 8))
	names =  [
		'big_bottom',
		'big_top',
		'big_right',
		'big_back',
		'big_left',
		'big_front',
		'small_bottom',
		'small_top',
		'small_right',
		'small_back',
		'small_left',
		'small_front',		
		]
	return pd.DataFrame(faces, index=names, columns=['i','j', 'k','l'])



class TwoBox:

	def __init__(self, xyz, triangles, edges):
		self.xyz = xyz
		self.triangles = triangles 
		self.edges = edges

	def show(self, edges=True):
		fig, ax = plt.subplots(1,1, subplot_kw={"projection":"3d"})
		ax.set_aspect('equal')
		for i in range(self.xyz.shape[0]):
			ax.text(*self.xyz[i], i, color='blue')
		if edges is not False:
			for edge in self.edges:
				x = self.xyz[[edge[0], edge[1]], 0]
				y = self.xyz[[edge[0], edge[1]], 1]
				z = self.xyz[[edge[0], edge[1]], 2]
				ax.plot3D(x,y,z, color='blue')
		ax.set_xlabel("x")
		ax.set_ylabel("y")
		ax.set_zlabel("z")
		fig.show()

	@property
	def volume(self):
		A = self.xyz[self.triangles[:, 0]]
		B = self.xyz[self.triangles[:, 1]]
		C = self.xyz[self.triangles[:, 2]]
		return np.abs(np.sum(np.vecdot(A, np.cross(B,C))) / 6)

	@property
	def surface_area(self):
		AB = self.xyz[self.triangles[:, 1]] - self.xyz[self.triangles[:, 0]]
		AC = self.xyz[self.triangles[:, 2]] - self.xyz[self.triangles[:, 0]]
		return np.sum(np.linalg.norm(np.cross(AB, AC), axis=1)) / 2


class TwoBoxFamily:
	def __init__(self, width_x, height_y, depth_z, angles, rng):
		j = get_position_dct()
		normals = make_normals(angles)
		extreme_verts = identify_extreme_vertices(normals)
		equality_names = ["y0", "y2", "y8", "y10"] + [extreme_verts[key] for key in ['x_max', 'y_max', 'z_max', 'z_min']]
		equality_values = [0  ,    0,    0,     0] + [width_x/2, height_y, depth_z, 0]
		lower = ['z2', 'z10', 'y8','y10', 'x8', 'y12', 'y14'] + [width_x/2*.1]*8
		upper = ['z0', 'z2', 'y12', 'y14', 'x2', 'y6', 'y6'] + [f'x{i}' for i in range(0,16,2)]
		faces_right = {
			"big_right":[0,2,4,6], 
			"big_top":[4,6],
			"big_front":[2,6,8,12],
			"big_back":[0,4],
			"small_top":[12,14],
			"small_front":[10,14],
			"small_right":[8,10,12,14]
		}
		A_eq, b_eq = _build_matrix_Ab(normals, faces_right, equality_names, equality_values, j)
		A_ub, b_ub = _build_matrix_Cd(lower, upper, j, width_x, height_y, depth_z)
		Q = _build_matrix_Q(j)
		A_eq_pinv = np.linalg.pinv(A_eq.toarray())
		u,s,vt = np.linalg.svd(np.eye(len(A_eq_pinv)) - A_eq_pinv @ A_eq)
		self.U = u[:, s > 1e-8]
		self.dim_null_space = self.U.shape[1]
		# u,s,vt = np.linalg.svd(A_eq.toarray())
		# m,n = A_eq.shape
		# self.U = vt[m-n:, :].T
		# self.dim_null_space = abs(m - n)
		# s_plus = np.zeros_like(s)
		# s_plus[s > 0] = 1 / s[s > 0]
		# S_plus = np.zeros((n,m))
		# S_plus[np.arange(m), np.arange(m)] = s_plus
		# A_eq_pinv = vt.T @ S_plus @ u.T 
		self.A_pinv_b = A_eq_pinv @ b_eq
		self.C = A_ub @ self.U
		self.d = b_ub - A_ub @ A_eq_pinv @ b_eq
		self.P = self.U.T @ Q @ self.U
		self.q = 2 * b_eq.T @ A_eq_pinv.T @ Q @ self.U
		self.r = b_eq.T @ A_eq_pinv.T @ Q @ A_eq_pinv @ b_eq
		possible = check_constraints_possible(len(self.P), self.P, self.q, self.r, self.C, self.d)
		if not possible["soln_exists"]:
			raise ValueError("Impossible Shape Requested!")
		self.p = pc.Polytope(self.C, self.d)
		self.extreme = pc.extreme(self.p)
		self.rng = rng
		self.triangles, self.edges = get_triangles_edges()
		self.x0 = possible['solution']

	def sample(self, n=1):
		cond1 = np.allclose(self.P, 0)
		cond2 = np.all(np.squeeze(self.extreme[:, None,:] @ self.P @ self.extreme[:, :, None]) + np.vecdot(self.extreme, self.q) + self.r <= 0)
		if cond1:
			C = np.vstack((self.C, self.q))
			d = np.hstack((self.d, -self.r))
			self.p = pc.Polytope(C,d)
			self.extreme = pc.extreme(self.p)
			extreme_weights = random_weight_uniform(
				size=(n, len(self.extreme)), 
				rng=self.rng)
			w_samples = extreme_weights @ self.extreme
		elif cond2:
			extreme_weights = random_weight_uniform(
				size=(n, len(self.extreme)), 
				rng=self.rng)
			w_samples = extreme_weights @ self.extreme
		else:
			w_samples = hit_and_run(n, self.x0, self.P, self.q, self.r, self.C, self.d, self.rng)

		samples = []
		for i in range(len(w_samples)):
			possible_solution = self.A_pinv_b + self.U @ w_samples[i]
			xyz =  np.full((16,3), np.nan)
			xyz[::2] = possible_solution[:24].reshape(-1,3)
			xyz[1::2] = xyz[::2] * np.array([-1,1,1])
			samples.append(TwoBox(xyz, self.triangles, self.edges))
		if len(w_samples) == 1:
			return samples[0]
		else:
			return samples


def format_tris_for_pyvista(triangles):
	face_edges = 3 * np.ones((len(triangles), 1), dtype = np.int32)
	return np.ravel(np.hstack((face_edges, triangles)))


def show_solid(xyz, triangles, edges=None):
	pv_triangles = format_tris_for_pyvista(triangles)
	mesh = pv.PolyData(xyz, pv_triangles)
	pl = pv.Plotter()
	pl.add_mesh(mesh, color = 'silver')
	if edges is not None:
		for i,j in edges:
			cylinder = pv.Cylinder(
			    center=(xyz[i] + xyz[j])/ 2, direction=xyz[j] - xyz[i], 
			    radius=.02, height=np.linalg.norm(xyz[j] - xyz[i])
			)
			pl.add_mesh(cylinder, color = 'black')
	pl.show()


def constraint_writer(width_x, height_y, depth_z, angles):
	j = get_position_dct()
	normals = make_normals(angles)
	extreme_verts = identify_extreme_vertices(normals)
	equality_names = ["y0", "y2", "y8", "y10"] + [extreme_verts[key] for key in ['x_max', 'y_max', 'z_max', 'z_min']]
	equality_values = [0  ,    0,    0,     0] + [width_x/2, height_y, depth_z, 0]
	lower = ['z2', 'z10', 'y8','y10', 'x8', 'y12', 'y14'] + [width_x/2*.1]*8
	upper = ['z0', 'z2', 'y12', 'y14', 'x2', 'y6', 'y6'] + [f'x{i}' for i in range(0,16,2)]
	faces_right = {
		"big_right":[0,2,4,6], 
		"big_top":[4,6],
		"big_front":[2,6,8,12],
		"big_back":[0,4],
		"small_top":[12,14],
		"small_front":[10,14],
		"small_right":[8,10,12,14]
	}
	A_eq, b_eq = _build_matrix_Ab(normals, faces_right, equality_names, equality_values, j)
	A_ub, b_ub = _build_matrix_Cd(lower, upper, j, width_x, height_y, depth_z)
	Q = _build_matrix_Q(j)
	jinv = {v:k for k,v in j.items()}
	A = A_eq.toarray()
	print("-----------------------")
	print("EQUALITY CONSTRAINTS)")
	print("-----------------------")
	for i in range(len(A)):
		lst = []
		for k in range(A.shape[1]):
			value = A[i,k]
			if value != 0:
				lst.append(f"{value:.03f} * {jinv[k]}")
		s = " + ".join(lst) + " = " + str(b_eq[i])
		print(s)
	print("-----------------------")
	print("INEQUALITY CONSTRAINTS")
	print("-----------------------")
	A = A_ub.toarray()
	for i in range(len(A)):
		lst = []
		for k in range(A.shape[1]):
			value = A[i,k]
			if value != 0:
				lst.append(f"{value:.03f} * {jinv[k]}")
		s = " + ".join(lst) + " <= " + f"{b_ub[i]:.03f}"
		print(s)
	print("-----------------------")
	print("QUADRATIC CONSTRAINT")
	print("-----------------------")
	lst = []
	visited = set()
	for i in range(len(Q)):
		for k in range(Q.shape[1]):
			value = Q[i,k]
			if value != 0 and ((k,i) not in visited):
				lst.append(f"{value:.03f} * {jinv[k]} * {jinv[i]}")
				visited.add((i,k))
	s = " + ".join(lst) + " <= " + str(0)
	print(s)


if __name__ == "__main__":

	rng = np.random.default_rng(seed=1234)
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
	show_solid(obj.xyz, obj.triangles, edges=obj.edges)


	# ## Can also import off files with pyvista instead of open3d
	# mesh = pv.read("./../../../dropbox/rigidity/ModelNet40/Car/train/car_0015.off")
	# mesh.plot(color ='silver', smooth_shading=False)


	# constraint_writer(7, 7, 7, angle_dct)



