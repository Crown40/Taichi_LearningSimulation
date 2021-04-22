import taichi as ti
import numpy as np
ti.init(arch=ti.gpu) # Try to run on GPU
dim=3
size=1
E_scale=1
quality = 1 # Use a larger value for higher-res simulations
n_particles, n_grid = 6000,128 #9000 * ,128 * quality #quality**2, 128 * quality
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 1e-4 / quality
p_vol, p_rho = (dx * 0.5)**dim, 1
p_mass = p_vol * p_rho
E, nu = 7.5e3*size*E_scale,0.2 #0.1e4, 0.2 # Young's modulus and Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1+nu) * (1 - 2 * nu)) # Lame parameters
x = ti.Vector.field(dim, dtype=float, shape=n_particles) # position
v = ti.Vector.field(dim, dtype=float, shape=n_particles) # velocity
C = ti.Matrix.field(dim, dim, dtype=float, shape=n_particles) # affine velocity field
F = ti.Matrix.field(dim, dim, dtype=float, shape=n_particles) # deformation gradient
material = ti.field(dtype=int, shape=n_particles) # material id
Jp = ti.field(dtype=float, shape=n_particles) # plastic deformation
grid_v = ti.Vector.field(dim, dtype=float, shape=(n_grid,)*dim) # grid node momentum/velocity
grid_m = ti.field(dtype=float, shape=(n_grid,)*dim) # grid node mass

@ti.kernel
def substep():
  for I in ti.grouped(grid_m):
    grid_v[I] = ti.Matrix.zero(int,dim) #[0, 0]
    grid_m[I] = 0
  for p in x: # Particle state update and scatter to grid (P2G)
    base = (x[p] * inv_dx - 0.5).cast(int)
    fx = x[p] * inv_dx - base.cast(float)
    # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
    w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
    F[p] = (ti.Matrix.identity(float, dim) + dt * C[p]) @ F[p] # deformation gradient update
    h = ti.exp(10 * (1.0 - Jp[p])) # Hardening coefficient: snow gets harder when compressed
    if material[p] == 2: #1 jelly, make it softer
      h = 0.3
    mu, la = mu_0 * h, lambda_0 * h
    if material[p] == 1: #0 liquid
      mu = 0.0
    U, sig, V = ti.svd(F[p])
    J = 1.0
    for d in ti.static(range(dim)):
      new_sig = sig[d, d]
      if material[p] == 0:  #2 Snow
        new_sig = min(max(sig[d, d], 1 - 2.5e-2), 1 + 4.5e-3)  # Plasticity
      Jp[p] *= sig[d, d] / new_sig
      sig[d, d] = new_sig
      J *= new_sig
    if material[p] == 1:  #0 Reset deformation gradient to avoid numerical instability
        #F[p] = ti.Matrix.identity(float, dim) * #ti.sqrt(J)
        new_F=ti.Matrix.identity(float, dim)
        new_F[0,0]=J
        F[p]=new_F
    elif material[p] == 0:#2
      F[p] = U @ sig @ V.transpose() # Reconstruct elastic deformation gradient after plasticity
    stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose() + ti.Matrix.identity(float, dim) * la * J * (J - 1)
    stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
    affine = stress + p_mass * C[p]
    for i, j, k in ti.static(ti.ndrange(*[3,]*dim)): # Loop over 3x3 grid node neighborhood
      offset = ti.Vector([i, j, k])
      dpos = (offset.cast(float) - fx) * dx
      weight = w[i][0] * w[j][1] * w[k][2]
      grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
      grid_m[base + offset] += weight * p_mass
  for I in ti.grouped(grid_m):
    if grid_m[I] > 0: # No need for epsilon here
      grid_v[I] = (1 / grid_m[I]) * grid_v[I] # Momentum to velocity
      grid_v[I][1] -= dt * 9.8 #50 # gravity
      for c in ti.static(range(dim)):
        if I[c] < 3 and grid_v[I][c] < 0:          grid_v[I][c] = 0 # Boundary conditions
        if I[c] > n_grid - 3 and grid_v[I][c] > 0: grid_v[I][c] = 0
      #if i < 3 and grid_v[I][0] < 0:          grid_v[I][0] = 0 # Boundary conditions
      #if i > n_grid - 3 and grid_v[I][0] > 0: grid_v[I][0] = 0
      #if j < 3 and grid_v[I][1] < 0:          grid_v[I][1] = 0
      #if j > n_grid - 3 and grid_v[I][1] > 0: grid_v[I][1] = 0
  for p in x: # grid to particle (G2P)
    base = (x[p] * inv_dx - 0.5).cast(int)
    fx = x[p] * inv_dx - base.cast(float)
    w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
    new_v = ti.Vector.zero(float, dim)
    new_C = ti.Matrix.zero(float, dim, dim)
    for i, j, k  in ti.static(ti.ndrange(*[3,]*dim)): # loop over 3x3 grid node neighborhood
      dpos = ti.Vector([i, j, k]).cast(float) - fx
      g_v = grid_v[base + ti.Vector([i, j, k])]
      weight = w[i][0] * w[j][1] * w[k][2]
      new_v += weight * g_v
      new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
    v[p], C[p] = new_v, new_C
    x[p] += dt * v[p] # advection

group_size = n_particles #// 2
@ti.kernel
def initialize():
  for i in range(n_particles):
    x[i] = [ti.random() * 0.2 + 0.3 + 0.0 * (i // group_size), ti.random() * 0.2 + 0.05 + 0.3 * (i // group_size), ti.random() * 0.2 + 0.3 + 0.0 * (i // group_size)]
    material[i] = i // group_size # 0: fluid 1: jelly 2: snow
    v[i] = ti.Matrix.zero(int,dim) #ti.Matrix([0, 0])
    F[i] = ti.Matrix.identity(int,dim) #ti.Matrix([[1, 0], [0, 1]])
    Jp[i] = 1

def T(a):
    if dim == 2:
        return a

    phi, theta = np.radians(28), np.radians(32)

    a = a - 0.5
    x, y, z = a[:, 0], a[:, 1], a[:, 2]
    c, s = np.cos(phi), np.sin(phi)
    C, S = np.cos(theta), np.sin(theta)
    x, z = x * c + z * s, z * c - x * s
    u, v = x, y * C + z * S
    return np.array([u, v]).swapaxes(0, 1) + 0.5

from simulation_modeling.marching_isovalue_advance import McIsovalue
marchingCube=McIsovalue(dim,64,1.0,True,False)

initialize()
gui = ti.GUI("Taichi MLS-MPM-99", res=512, background_color=0x112F41)
file_name='example\ply\Snow.ply'
time_step=int(8e-3 // dt)
colors = np .array([0xEEEEF0,0x55ED3B, 0x068587], dtype=np.uint32)
color= np .array([ [0x55/256,0xED/256,0x3B/256], [0x06/256,0x85/256,0x87/256], [0xEE/256,0xEE/256,0xF0/256]], dtype=np.float)
#while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
for frame in range(150):
  for s in range(time_step):
    substep()
  '''
  marchingCube.marching_cube(x,5,0,0)
  vertices_amount=min(marchingCube.vertices_amount[None],marchingCube.vertices_datasize)
  print(frame,'vertices_amount:',vertices_amount)
  np_pos=marchingCube.vertex_pos.to_numpy()[:vertices_amount]
  np_normal=marchingCube.vertex_normal.to_numpy()[:vertices_amount]
  np_color=color[np.ones(vertices_amount,int)]
  writer=ti.PLYWriter(vertices_amount,vertices_amount//3)
  writer.add_faces(np.arange(vertices_amount))
  writer.add_vertex_pos(np_pos[:,0],np_pos[:,1],np_pos[:,2])
  writer.add_vertex_normal(np_normal[:,0],np_normal[:,1],np_normal[:,2])
  writer.add_vertex_color(np_color[:,0],np_color[:,1],np_color[:,2])
  writer.export_frame_ascii(frame,file_name)
  '''
  #
  pos=x.to_numpy()
  if file_name:
        writer = ti.PLYWriter(num_vertices=n_particles)
        writer.add_vertex_pos(pos[:, 0], pos[:, 1], pos[:, 2])
        writer.export_frame(frame, file_name)
  #print(frame,pos)
  gui.circles(T(pos), radius=1.5, color=colors[material.to_numpy()])
  gui.show() # Change to gui.show(f'{frame:06d}.png') to write images to disk

