import taichi as ti
import numpy as np
import math
ti.init(arch=ti.cpu) # Try to run on GPU
dim=3
space_length=1.0
size=1
E_scale=1
quality = 1 # Use a larger value for higher-res simulations
n_particles, n_grid = 2*16384,256 #8*16384,512  #6000,128 #9000 * ,128 * quality #quality**2, 128 * quality
dx, inv_dx = space_length / n_grid, float(n_grid)
dt = 1e-5 / quality #1e-5
p_vol, p_rho = (dx * 0.5)**dim, 4e2 #1.0  #4e2  
p_mass = p_vol * p_rho
E, nu = 1.4e5*size*E_scale,0.2 #0.1e4, 0.2 # Young's modulus and Poisson's ratio 1.4e5
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1+nu) * (1 - 2 * nu)) # Lame parameters
x = ti.Vector.field(dim, dtype=float, shape=n_particles) # position
v = ti.Vector.field(dim, dtype=float, shape=n_particles) # velocity
C = ti.Matrix.field(dim, dim, dtype=float, shape=n_particles) # affine velocity field
F = ti.Matrix.field(dim, dim, dtype=float, shape=n_particles) # deformation gradient
material = ti.field(dtype=int, shape=n_particles) # material id
Jp = ti.field(dtype=float, shape=n_particles) # plastic deformation

#grid_v = ti.Vector.field(dim, dtype=float, shape=(n_grid,)*dim) # grid node momentum/velocity
#grid_m = ti.field(dtype=float, shape=(n_grid,)*dim) # grid node mass
g_indices=[ti.ij,ti.ijk][dim-2]
grid_space=ti.root.pointer(g_indices,int(n_grid//32))
grid_v = ti.Vector.field(dim, dtype=float)
grid_m = ti.field(dtype=float)
grid_space.dense(g_indices,32).place(grid_m,grid_v)
#
box_bound=3 #3
zero_error=1e-7
# p
xi=10.0 #10.0
critical_compression,critical_stretch=2.5e-2,4.5e-3 #2.5e-2,4.5e-3 
# g
g_acc=1e3 #1e3  #9.8
vCO=ti.Vector([0,]*dim)
# collision grid
#status_g=ti.field(int,(n_grid,)*dim)
#normal_g=ti.Vector.field(dim,float,(n_grid,)*dim)
status_g=ti.field(int)
normal_g=ti.Vector.field(dim,float)
ti.root.pointer(g_indices,int(n_grid//32)).bitmasked(g_indices,32).place(status_g,normal_g)
x_ndrange,y_ndrange,z_ndrange=[[1,n_grid],[1,n_grid,n_grid]][dim-2],[[n_grid,1],[n_grid,1,n_grid]][dim-2],[[0,0],[n_grid,n_grid,1]][dim-2]
edge_length=0.2*space_length
steps1=int(math.ceil(edge_length*math.sqrt(2)*0.5*inv_dx))+1
steps2=int(math.ceil(edge_length*inv_dx))+1
vertex_pos=ti.Vector.field(dim,float,6)
face_index=ti.field(int,8*3)

@ti.kernel
def substep():
  friction_coef=10.0 #-mu_0*ti.exp(xi) #10.0 #-10.0
  #for I in ti.grouped(grid_m):
  #  grid_v[I] = ti.Matrix.zero(int,dim) #[0, 0]
  #  grid_m[I] = 0
  for p in x: # Particle state update and scatter to grid (P2G)
    base = (x[p] * inv_dx - 0.5).cast(int)
    fx = x[p] * inv_dx - base.cast(float)
    # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
    w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
    F[p] = (ti.Matrix.identity(float, dim) + dt * C[p]) @ F[p] # deformation gradient update
    h = ti.exp(xi * (1.0 - Jp[p])) # Hardening coefficient: snow gets harder when compressed
    if material[p] == 2: #1 jelly, make it softer
      h = 0.3
    mu, la = mu_0 * h, lambda_0 * h
    #friction_coef=mu
    #friction_coef=mu_0*ti.exp(xi) #*10.0
    if material[p] == 1: #0 liquid
      mu = 0.0
    U, sig, V = ti.svd(F[p])
    J = 1.0
    for d in ti.static(range(dim)):
      new_sig = sig[d, d]
      if material[p] == 0:  #2 Snow
        new_sig = min(max(sig[d, d], 1 - critical_compression), 1 + critical_stretch)  # Plasticity
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
      vG = (1 / grid_m[I]) * grid_v[I] # Momentum to velocity
      vG[1] -= dt * g_acc #50 # gravity
      if status_g[I] > 0:
          normal=normal_g[I]
          #vG=[0,]*dim
          #vG-=normal*normal.dot(vG) #vG-=normal*ti.min(normal.dot(vG),0)  #vG=[0,0,0]
          vRel=vG-vCO
          vN=vRel.dot(normal)
          if vN < 0:
              vT=vRel-normal*vN
              vT_norm=vT.norm(zero_error)
              result=-friction_coef*vN
              if(vT_norm<=result): # no collision
                vRel=ti.Vector.zero(float,dim)
              else: # 
                vRel=vT+1/vT_norm*vT*friction_coef*vN
              vG=vRel+vCO
      '''
      '''
      
      for c in ti.static(range(dim)):
        if I[c] < box_bound :   vG=[0,]*dim
        elif I[c] > n_grid - box_bound: vG=[0,]*dim
        
      
      grid_v[I]=vG
      '''
      '''
      
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
interval=0.2*space_length/32
amount=ti.field(int,())
@ti.kernel
def initialize():
  for i in range(n_particles):
  #for I in ti.grouped(ti.ndrange(64,32,16)):
  #  for d in ti.static(range(4)):
        #i=ti.atomic_add(amount[None],1)
        x[i]=ti.Vector([0.35+0.3*ti.random(),0.4+0.2*ti.random(),0.425+0.15*ti.random()])*space_length
        #x[i]=I*interval+[0.3,0.3,0.45]
        #x[i]=[0.45+ti.random()*0.1 for _  in ti.static(range(dim))]
        #x[i]=I*interval+[0.45+ti.random()*interval for _ in ti.static(range(dim))]
        #i=ti.static(a*1024+b*64+c*4+d)
        #x[i] = [ti.random() * 0.2 + 0.3 + 0.0 * (i // group_size), ti.random() * 0.2 + 0.05 + 0.3 * (i // group_size), ti.random() * 0.2 + 0.3 + 0.0 * (i // group_size)]
        #x[i]=[0.45+ti.random()*0.1 for _ in ti.static(range(dim))]
        #material[i] = i // group_size # 0: fluid 1: jelly 2: snow
        #x[i]=[0.45+ti.random()*0.1,0.1+ti.random()*0.1,0.45+ti.random()*0.1]
        v[i] = ti.Matrix.zero(int,dim) #-20.0*ti.Matrix.unit(dim,ti.static(0),float) #ti.Matrix.zero(int,dim) #ti.Matrix([0, 0])
        F[i] = ti.Matrix.identity(float,dim) #ti.Matrix([[1, 0], [0, 1]])
        Jp[i] = 1
  #
  if(ti.static(dim==3)):
        #
        # Cube-Bound
        cube_pos=ti.Vector([0.5,0.2,0.5])*space_length
        top_pos,left_pos,right_pos=cube_pos+ti.Vector([0,ti.sqrt(2)*0.5,0.5])*edge_length,cube_pos+ti.Vector([ti.sqrt(2)*0.5,0,0.5])*edge_length,cube_pos+ti.Vector([-ti.sqrt(2)*0.5,0,0.5])*edge_length
        for i in ti.static(range(2)):
            displacement=ti.Vector([0,0,-i])*edge_length
            vertex_pos[i]=top_pos+displacement
            vertex_pos[2+i]=left_pos+displacement
            vertex_pos[4+i]=right_pos+displacement
        face_index[0],face_index[1],face_index[2]=0,4,2
        face_index[3],face_index[4],face_index[5]=1,3,5
        face_index[6],face_index[7],face_index[8]=0,1,5
        face_index[9],face_index[10],face_index[11]=0,5,4
        face_index[12],face_index[13],face_index[14]=1,0,2
        face_index[15],face_index[16],face_index[17]=1,2,3
        face_index[18],face_index[19],face_index[20]=2,5,3
        face_index[21],face_index[22],face_index[23]=2,4,5
        norm1,norm2=(top_pos-left_pos).normalized(zero_error),(top_pos-right_pos).normalized(zero_error)
        dir1,dir2,dir3=(right_pos-top_pos)*(1/steps1),(left_pos-top_pos)*(1/steps1),ti.Vector([0,0,-edge_length])*(1/steps2)
        for i,j in ti.ndrange(*[steps1,steps2]):
                I=int((top_pos+i*dir1+j*dir3)*inv_dx)
                status_g[I]+=1
                normal_g[I]+=norm1
                for times in ti.static(range(box_bound-1)):
                    I[1]+=1
                    status_g[I]+=1
                    normal_g[I]+=norm1
        for i,j in ti.ndrange(*[steps1,steps2]):
                I=int((top_pos+i*dir2+j*dir3)*inv_dx)
                status_g[I]+=1
                normal_g[I]+=norm2
                for times in ti.static(range(box_bound-1)):
                    I[1]+=1
                    status_g[I]+=1
                    normal_g[I]+=norm2
        #
        for I in ti.grouped(status_g):
            if(status_g[I]>1):
                status_g[I]=1
                normal_g[I]=normal_g[I].normalized(zero_error)
        #
        #for I in ti.grouped(status_g):
        #    if(status_g[I]>0):
        #        J=I
        #        for times in ti.static(range(box_bound-1)):
        #            J[1]+=1
        #            normal_g[J]=normal_g[I]
        '''
        '''
  '''
  '''
      

def T(a):
    if dim == 2:
        return a

    phi, theta = np.radians(28), np.radians(32)
    offset=0.5*space_length #0.5

    #print(a)
    a = a - offset
    x, y, z = a[:, 0], a[:, 1], a[:, 2]
    c, s = np.cos(phi), np.sin(phi)
    C, S = np.cos(theta), np.sin(theta)
    x, z = x * c + z * s, z * c - x * s
    u, v = x, y * C + z * S
    return (np.array([u, v]).swapaxes(0, 1) + offset)*(1/space_length)


initialize()
gui = ti.GUI("Taichi MLS-MPM-99", res=512, background_color=0x112F41)
file_name='demo\\snowCube_p\\LowStretch\\LowStretch_p.ply'
cube_file='demo\\cube\\cube.ply'
time_step=50 #int(8e-3 // dt)
colors = np .array([0xEEEEF0,0x55ED3B, 0x068587], dtype=np.uint32)
color= np .array([[0x55/256,0xED/256,0x3B/256], [0xEE/256,0xEE/256,0xF0/256], [0x06/256,0x85/256,0x87/256]] , dtype=np.float32)
np_color=color[np.ones(n_particles,np.int32)]
#while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
for frame in range(241):
  for s in range(time_step):
    substep()
    grid_space.deactivate_all()
  print(frame)
  #
  pos=x.to_numpy()
  if file_name:
        '''
        cube_pos=vertex_pos.to_numpy()
        cube_index=face_index.to_numpy()
        writer=ti.PLYWriter(6,8)
        writer.add_vertex_pos(cube_pos[:,0],cube_pos[:,1],cube_pos[:,2])
        writer.add_faces(cube_index)
        writer.export_frame(frame,cube_file)
        '''
        writer = ti.PLYWriter(num_vertices=n_particles)
        writer.add_vertex_pos(pos[:, 0], pos[:, 1], pos[:, 2])
        writer.add_vertex_color(np_color[:, 0], np_color[:, 1], np_color[:, 2])
        writer.export_frame(frame, file_name)

  #print(frame,pos)
  gui.circles(T(pos), radius=1.5, color=colors[material.to_numpy()])
  gui.show() # Change to gui.show(f'{frame:06d}.png') to write images to disk

