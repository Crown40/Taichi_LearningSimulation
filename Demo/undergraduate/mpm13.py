import taichi as ti
import numpy as np
import math
export_folder='demo\\snowCube_p'
file_seq=0
export_file='demo\\snowCube_p\\lowerYoung_p\\lowerYoung_p.ply'
ti.init(ti.cpu)
# Misc:(杂项)MISCELLANEOUS
dim=3
n_particle=8192*2
n_grid=128
amount_grid=n_grid**dim
dx=1/n_grid
inv_dx=1/dx
inv_h=1.0
bspline_range=4
box_bound=3
zero_error=1e-7
alpha=0.95
#
dt=5e-5 #5e-4
time_steps=30
g=50.0
gravity_acc=ti.Vector([[0,-g],[0,-g,0]][dim-2])
vCO=ti.Vector([0,]*dim) # staic:[0,]*dim; dynamic:[?,]
# particle
densityP=4e2
mP=densityP*((dx*0.5)**dim) #4e2*(dx)**dim
print('mP:',mP)
E,nu= 4.8e4,0.2#4.8e5,0.2
mu_0,lambda_0=E/(2*(1+nu)),E*nu/((1+nu)*(1-2*nu))
mu,lambd=mu_0,lambda_0
#mu,lambd=mu_0,lambda_0
xi=10.0 #5.0
critical_compression,critical_stretch=2.5e-2,7.5e-3 #1.9e-2,7.5e-3 #2.5e-2,7.5e-3
    #
x_p=ti.Vector.field(dim,float,n_particle)
v_p=ti.Vector.field(dim,float,n_particle)
V_p=ti.field(float,n_particle)
F_E_p=ti.Matrix.field(dim,dim,float,n_particle)
F_P_p=ti.Matrix.field(dim,dim,float,n_particle)
#J_E_p=ti.field(float,n_particle)
#J_P_p=ti.field(float,n_particle)
# grid
m_g=ti.field(float,(n_grid,)*dim)
v_g=ti.Vector.field(dim,float,(n_grid,)*dim)
v_next_g=ti.Vector.field(dim,float,(n_grid,)*dim)
f_g=ti.Vector.field(dim,float,(n_grid,)*dim)
# Collision Surface
status_g=ti.field(int,(n_grid,)*dim)
normal_g=ti.Vector.field(dim,float,(n_grid,)*dim)
x_ndrange,y_ndrange,z_ndrange=[[1,n_grid],[1,n_grid,n_grid]][dim-2],[[n_grid,1],[n_grid,1,n_grid]][dim-2],[[0,0],[n_grid,n_grid,1]][dim-2]
edge_length=0.2
steps=int(edge_length*math.sqrt(6)/3*inv_dx)

@ti.kernel
def init(x_p:ti.template(),v_p:ti.template(),F_E_p:ti.template(),F_P_p:ti.template()): #,J_E_p:ti.template(),J_P_p:ti.template()):
    # Particle
    #interval=0.2/16
    #for i,j,k in ti.ndrange(16,16,16):
        #P=i*256+j*16+k
        #x_p[P]=0.3+ti.Vector([i,j,k])*interval
    for P in ti.grouped(x_p):
        x_p[P]=[0.2+ti.random()*0.2 for _ in ti.static(range(dim))]
        v_p[P]=ti.Vector.zero(float,dim)
        F_E_p[P]=ti.Matrix.identity(float,dim)
        F_P_p[P]=ti.Matrix.identity(float,dim)
        #J_E_p[P]=1
        #J_P_p[P]=1
    # Grid
        # Box-Bound
    if(ti.static(dim==3)):
        d=ti.static(0)
        n=ti.Vector.unit(dim,d,float)
        for I in ti.grouped(ti.ndrange(*x_ndrange)):
            for c in ti.static(range(box_bound)):
                    I[d]=c
                    status_g[I]+=1
                    normal_g[I]+=n
                    I[d]=n_grid-c
                    status_g[I]+=1
                    normal_g[I]-=n
        d=ti.static(1)
        n=ti.Vector.unit(dim,d,float)
        for I in ti.grouped(ti.ndrange(*y_ndrange)):
            for c in ti.static(range(box_bound)):
                    I[d]=c
                    status_g[I]+=1
                    normal_g[I]+=n
                    I[d]=n_grid-c
                    status_g[I]+=1
                    normal_g[I]-=n
        d=ti.static(2)
        n=ti.Vector.unit(dim,d,float)
        for I in ti.grouped(ti.ndrange(*z_ndrange)):
            for c in ti.static(range(box_bound)):
                    I[d]=c
                    status_g[I]+=1
                    normal_g[I]+=n
                    I[d]=n_grid-c
                    status_g[I]+=1
                    normal_g[I]-=n
        #
        # Cube-Bound
        cube_pos=ti.Vector([0.3,0.1,0.3])
        top_pos=cube_pos+ti.Vector([0,ti.sqrt(3)/3,0])*edge_length
        pos1,pos2,pos3=cube_pos+ti.Vector([0,0,ti.sqrt(6)/3])*edge_length,cube_pos+ti.Vector([ti.sqrt(2)/2,0,-ti.sqrt(6)/6])*edge_length,cube_pos+ti.Vector([-ti.sqrt(2)/2,0,-ti.sqrt(6)/6])*edge_length
        norm1,norm2,norm3=(top_pos-pos1).normalized(zero_error),(top_pos-pos2).normalized(zero_error),(top_pos-pos3).normalized(zero_error)
        dir1,dir2,dir3=(pos1-top_pos)*(1/steps),(pos2-top_pos)*(1/steps),(pos3-top_pos)*(1/steps)
        for i,j in ti.ndrange(*[steps,]*(dim-1)):
                I=int((top_pos+i*dir1+j*dir2)*inv_dx)
                status_g[I]+=1
                normal_g[I]+=norm2
                #for times in ti.static(range(1,3)):
                #    I[1]+=1
                #    status_g[I]+=1
                #    normal_g[I]+=norm3
        for i,j in ti.ndrange(*[steps,]*(dim-1)):
                I=int((top_pos+i*dir2+j*dir3)*inv_dx)
                status_g[I]+=1
                normal_g[I]+=norm1
                #for times in ti.static(range(1,3)):
                #    I[1]+=1
                #    status_g[I]+=1
                #    normal_g[I]+=norm1
        for i,j in ti.ndrange(*[steps,]*(dim-1)):
                I=int((top_pos+i*dir1+j*dir3)*inv_dx)
                status_g[I]+=1
                normal_g[I]+=norm2
                #for times in ti.static(range(1,3)):
                #    I[1]+=1
                #    status_g[I]+=1
                #    normal_g[I]+=norm2
        '''
        '''
        for I in ti.grouped(status_g):
            if(status_g[I]>1):
                status_g[I]=1
                normal_g[I]=normal_g[I].normalized(zero_error)

@ti.kernel
def reset(m_g:ti.template(),v_g:ti.template(),f_g:ti.template()): 
    for I in ti.grouped(m_g):
        m_g[I]=0.0
        v_g[I]=ti.Vector.zero(float,dim)
        f_g[I]=ti.Vector.zero(float,dim)

@ti.kernel
def P2G(mP:float,x_p:ti.template(),v_p:ti.template(),m_g:ti.template(),v_g:ti.template()):
    # p2g:scatter m,v :conserve mass and momentum
    for P in ti.grouped(x_p):
        xPG=x_p[P]*inv_dx
        Base=int(xPG-1.0)
        fx=xPG-float(Base)
        w=[-0.16666666666666666*fx**3+fx**2-2*fx+1.3333333333333333, 0.5*(fx-1)**3-(fx-1)**2+0.6666666666666666,
          0.5*(2-fx)**3-(2-fx)**2+0.6666666666666666, -0.16666666666666666*(3-fx)**3+(3-fx)**2-2*(3-fx)+1.3333333333333333]
        for Offset in ti.static(ti.grouped(ti.ndrange(*[bspline_range,]*dim))):
            weight=1.0
            for d in ti.static(range(dim)):
                weight*=w[Offset[d]][d]
            I=Base+Offset
            m_g[I]+=weight*mP
            v_g[I]+=weight*mP*v_p[P]
    for I in ti.grouped(m_g):
        if(m_g[I]!=0.0):
            v_g[I]*=1/m_g[I]

@ti.kernel
def computeVolumeNDensity(mP:float,x_p:ti.template(),V_p:ti.template(),m_g:ti.template()):
    # only once at first time
    inv_Vg=1/(dx)**3
    for P in ti.grouped(x_p):
        xPG=x_p[P]*inv_dx
        Base=int(xPG-1.0)
        fx=xPG-float(Base)
        w=[-0.16666666666666666*fx**3+fx**2-2*fx+1.3333333333333333, 0.5*(fx-1)**3-(fx-1)**2+0.6666666666666666,
          0.5*(2-fx)**3-(2-fx)**2+0.6666666666666666, -0.16666666666666666*(3-fx)**3+(3-fx)**2-2*(3-fx)+1.3333333333333333]
        for Offset in ti.static(ti.grouped(ti.ndrange(*[bspline_range,]*dim))):
            weight=1.0
            for d in ti.static(range(dim)):
                weight*=w[Offset[d]][d]
            # densitity:rho=ρ=Σ
            V_p[P]+=weight*m_g[Base+Offset]*inv_Vg
        # V=m/rho=m/ρ
        V_p[P]=1/V_p[P]*mP

@ti.kernel
def computeGridForce(x_p:ti.template(),F_E_p:ti.template(),F_P_p:ti.template(),f_g:ti.template()):
    for P in ti.grouped(x_p):
        xPG=x_p[P]*inv_dx
        Base=int(xPG-1.0)
        fx=xPG-float(Base)
        # PD: Polar Decomposition
        F_E=F_E_p[P]
        J_E=F_E.determinant() #J_E_p[P]
        # U,sigma,V=ti.svd(F_E)
        R_E,S_E=ti.polar_decompose(F_E)
        # Cauchy stress
        harden=ti.exp(xi*(1-F_P_p[P].determinant()) ) #J_P_p[P]))
        mu,lambd=mu_0*harden,lambda_0*harden
        stress=2*mu*(F_E - R_E) @ F_E.transpose() + ti.Matrix.identity(float, dim)*lambd*(J_E-1)*J_E
        # Volumn*Stress
        stress*=V_p[P]
        #
        w=[-0.16666666666666666*fx**3+fx**2-2*fx+1.3333333333333333, 0.5*(fx-1)**3-(fx-1)**2+0.6666666666666666,
          0.5*(2-fx)**3-(2-fx)**2+0.6666666666666666, -0.16666666666666666*(3-fx)**3+(3-fx)**2-2*(3-fx)+1.3333333333333333]
        grad_w=[-0.5*(fx**2)*inv_h+2*fx*inv_h-2, 1.5*((fx-1)**2)*inv_h-2*(fx-1)*inv_h,
               -1.5*((fx-2)**2)*inv_h-2*(fx-2)*inv_h, 0.5*((fx-3)**2)*inv_h+2*(fx-3)*inv_h+2]
        for Offset in ti.static(ti.grouped(ti.ndrange(*[bspline_range,]*dim))):
            grad_weight=ti.Vector.zero(float,dim)
            for c in ti.static(range(dim)):
                weight=1.0
                for d in ti.static(range(dim)):
                    weight*=w[Offset[d]][d] if(d!=c) else 1.0
                weight*=grad_w[Offset[c]][c]*inv_dx
                grad_weight[c]=weight
            f_g[Base+Offset]-=stress@grad_weight

@ti.kernel
def updateGridVelocity(m_g:ti.template(),v_next_g:ti.template(),v_g:ti.template(),f_g:ti.template()):
    for I in ti.grouped(m_g):
        if(m_g[I]!=0.0):
            v_next_g[I]=v_g[I]+dt*(1/m_g[I]*f_g[I]+gravity_acc)

@ti.kernel
def collideGridBody(m_g:ti.template(),v_next_g:ti.template(),status_g:ti.template(),normal_g:ti.template()):
    for I in ti.grouped(v_next_g):
        if(m_g[I]!=0 and status_g[I]>0):
            if_collision=1
            normal=normal_g[I]
            #vG=v_next_g[I]
            #normal=ti.Vector.zero(float,dim)
            #for d in ti.static(range(dim)):
            #    if(I[d]<ti.static(box_bound)):
            #        normal[d]=1.0
            #        vG[d]=0.0
            #    elif(I[d]>ti.static(n_grid-box_bound)):
            #        normal[d]=-1.0
            #        vG[d]=0.0
            #v_next_g[I]=vG
            if(if_collision):
                #normal.normalized(zero_error)
                vRel=v_next_g[I]-vCO
                vN=vRel.dot(normal)
                # vN:v_n>=0 <=> no collision
                # v_n<0 <=> yes collision
                if(vN<0):
                    vT=vRel-normal*vN
                    vT_norm=vT.norm(zero_error)
                    if(vT_norm<=-mu*vN): # no collision
                        vRel=ti.Vector.zero(float,dim)
                    else: # 
                        vRel=vT+1/vT_norm*vT*mu*vN
                # v'=v_rel^'+v_co
                v_next_g[I]=vRel+vCO

@ti.kernel
def integrateSemiImplicit():
    pass

@ti.kernel
def updateDeformationGradient(x_p:ti.template(),F_E_p:ti.template(),F_P_p:ti.template(),v_next_g:ti.template()):
    for P in ti.grouped(x_p):
        xPG=x_p[P]*inv_dx
        Base=int(xPG-1.0)
        fx=xPG-float(Base)
        # update F_E
        F_E=F_E_p[P]
        grad_v=ti.Matrix.zero(float,dim,dim)
        #
        w=[-0.16666666666666666*fx**3+fx**2-2*fx+1.3333333333333333, 0.5*(fx-1)**3-(fx-1)**2+0.6666666666666666,
          0.5*(2-fx)**3-(2-fx)**2+0.6666666666666666, -0.16666666666666666*(3-fx)**3+(3-fx)**2-2*(3-fx)+1.3333333333333333]
        grad_w=[-0.5*(fx**2)*inv_h+2*fx*inv_h-2, 1.5*((fx-1)**2)*inv_h-2*(fx-1)*inv_h,
               -1.5*((fx-2)**2)*inv_h-2*(fx-2)*inv_h, 0.5*((fx-3)**2)*inv_h+2*(fx-3)*inv_h+2]
        for Offset in ti.static(ti.grouped(ti.ndrange(*[bspline_range,]*dim))):
            grad_weight=ti.Vector.zero(float,dim)
            for c in ti.static(range(dim)):
                weight=1.0
                for d in ti.static(range(dim)):
                    weight*=w[Offset[d]][d] if(d!=c) else 1.0
                weight*=grad_w[Offset[c]][c]*inv_dx
                grad_weight[c]=weight
            grad_v+=v_next_g[Base+Offset].outer_product(grad_weight)
        F_E*=ti.Matrix.identity(float,dim)+dt*grad_v
        F=F_E@F_P_p[P]
        # svd
        U,Sigma,V=ti.svd(F_E)
        for d in ti.static(range(dim)):
            new_elem=Sigma[d,d]
            new_elem=min(max(Sigma[d,d],1-critical_compression),1+critical_stretch)
            Sigma[d,d]=new_elem
        F_E_p[P]=U@Sigma@V.transpose()
        F_P_p[P]=V@Sigma.inverse()@U.transpose()@F

@ti.kernel
def updateParticleVelocity(x_p:ti.template(),v_p:ti.template(),v_next_g:ti.template(),v_g:ti.template()):
    for P in ti.grouped(x_p):
        xPG=x_p[P]*inv_dx
        Base=int(xPG-1.0)
        fx=xPG-float(Base)
        #
        v_PIC=ti.Vector.zero(float,dim)
        v_FLIP=ti.Vector.zero(float,dim)
        #
        w=[-0.16666666666666666*fx**3+fx**2-2*fx+1.3333333333333333, 0.5*(fx-1)**3-(fx-1)**2+0.6666666666666666,
          0.5*(2-fx)**3-(2-fx)**2+0.6666666666666666, -0.16666666666666666*(3-fx)**3+(3-fx)**2-2*(3-fx)+1.3333333333333333]
        for Offset in ti.static(ti.grouped(ti.ndrange(*[bspline_range,]*dim))):
            weight=1.0
            for d in ti.static(range(dim)):
                weight*=w[Offset[d]][d]
            I=Base+Offset
            v_PIC+=v_next_g[I]*weight
            v_FLIP+=(v_next_g[I]-v_g[I])*weight
        v_FLIP+=v_p[P]
        v_p[P]=(1-alpha)*v_PIC+alpha*v_FLIP

@ti.kernel
def collideParticleBody(x_p:ti.template(),v_p:ti.template(),status_g:ti.template(),normal_g:ti.template()):
    for P in ti.grouped(x_p):
        #vP=v_p[P]
        I=int(x_p[P]*inv_dx)
        if(status_g[I]>0):
            if_collision=1
            normal=normal_g[I]
            #normal=ti.Vector.zero(float,dim)
            #for d in ti.static(range(dim)):
            #    if(I[d]<ti.static(box_bound)):
            #        normal[d]=1.0
                    #if_collision=1
            #        vP[d]=0.0
            #    elif(I[d]>ti.static(n_grid-box_bound)):
            #        normal[d]=-1.0
                    #if_collision=1
            #        vP[d]=0.0
            #v_p[P]=vP
            if(if_collision):
                normal.normalized(zero_error)
                vRel=v_p[P]-vCO
                vN=vRel.dot(normal)
                # vN:v_n>=0 <=> no collision
                # <0 <=> yes collision
                if(vN<0):
                    vT=vRel-normal*vN
                    vT_norm=vT.norm(zero_error)
                    if(vT_norm<=-mu*vN): # no collision
                        vRel=ti.Vector.zero(float,dim)
                    else: # 
                        vRel=vT+1/vT_norm*vT*mu*vN
                v_p[P]=vRel+vCO


@ti.kernel
def updateParticlePosition(x_p:ti.template(),v_p:ti.template()):
    for P in ti.grouped(x_p):
        x_p[P]+=dt*v_p[P]


def firstStep():
    #reset(m_g,v_g,f_g)
    P2G(mP,x_p,v_p,m_g,v_g)
    computeVolumeNDensity(mP,x_p,V_p,m_g)
    #computeGridForce(x_p,F_E_p,F_P_p,f_g)
    #updateGridVelocity(m_g,v_next_g,v_g,f_g)
    #collideGridBody(m_g,v_next_g)
    #solve linear system: explict|semi-implicit
    #updateDeformationGradient(x_p,F_E_p,F_P_p,v_next_g)
    #updateParticleVelocity(x_p,v_p,v_next_g,v_g)
    #collideParticleBody(x_p,v_p)
    #updateParticlePosition(x_p,v_p)

def subStep():
    reset(m_g,v_g,f_g)
    P2G(mP,x_p,v_p,m_g,v_g)
    #computeVolumeNDensity(mP,x_p,V_p,m_g)
    computeGridForce(x_p,F_E_p,F_P_p,f_g)
    updateGridVelocity(m_g,v_next_g,v_g,f_g)
    collideGridBody(m_g,v_next_g,status_g,normal_g)
    #solve linear system: explict|semi-implicit
    updateDeformationGradient(x_p,F_E_p,F_P_p,v_next_g)
    updateParticleVelocity(x_p,v_p,v_next_g,v_g)
    collideParticleBody(x_p,v_p,status_g,normal_g)
    updateParticlePosition(x_p,v_p)

def debugSubStep():
    reset(m_g,v_g,f_g)
    P2G(mP,x_p,v_p,m_g,v_g)
    np.savetxt(export_folder+'\\P2G.v_p_'+str(file_seq)+'.txt',np.reshape(v_p.to_numpy(),(n_particle,dim)),delimiter=',' )
    np.savetxt(export_folder+'\\P2G.v_g_'+str(file_seq)+'.txt',np.reshape(v_g.to_numpy(),(a_grid,dim)),delimiter=',' )
    #print('P2G.v_p:',v_p)
    #print('P2G.v_g:',v_g)
    #computeVolumeNDensity(mP,x_p,V_p,m_g)
    computeGridForce(x_p,F_E_p,F_P_p,f_g)
    np.savetxt(export_folder+'\\CGF.f_g_'+str(file_seq)+'.txt',np.reshape(f_g.to_numpy(),(a_grid,dim)),delimiter=',' )
    #print('CGF.f_g:',f_g)
    updateGridVelocity(m_g,v_next_g,v_g,f_g)
    np.savetxt(export_folder+'\\UGV.v_next_g_'+str(file_seq)+'.txt',np.reshape(v_next_g.to_numpy(),(a_grid,dim)),delimiter=',' )
    #print('UGV.v_next_g:',v_next_g)
    collideGridBody(m_g,v_next_g)
    np.savetxt(export_folder+'\\CGB.v_next_g_'+str(file_seq)+'.txt',np.reshape(v_next_g.to_numpy(),(a_grid,dim)),delimiter=',' )
    #print('CGB.v_next_g:',v_next_g)
    #solve linear system: explict|semi-implicit
    updateDeformationGradient(x_p,F_E_p,F_P_p,v_next_g)
    np.savetxt(export_folder+'\\UDG.F_E_p_'+str(file_seq)+'.txt',np.reshape(F_E_p.to_numpy(),(n_particle*dim,dim)),delimiter=',' )
    np.savetxt(export_folder+'\\UDG.F_P_p_'+str(file_seq)+'.txt',np.reshape(F_P_p.to_numpy(),(n_particle*dim,dim)),delimiter=',' )
    #print('UDG.F_E_p:',F_E_p)
    #print('UDG.F_P_p:',F_P_p)
    updateParticleVelocity(x_p,v_p,v_next_g,v_g)
    np.savetxt(export_folder+'\\UPV.v_p_'+str(file_seq)+'.txt',np.reshape(v_p.to_numpy(),(n_particle,dim)),delimiter=',' )
    #print('UPV.v_p:',v_p)
    collideParticleBody(x_p,v_p)
    np.savetxt(export_folder+'\\CPB.v_p_'+str(file_seq)+'.txt',np.reshape(v_p.to_numpy(),(n_particle,dim)),delimiter=',' )
    #print('CPB.v_p:',v_p)
    updateParticlePosition(x_p,v_p)
    np.savetxt(export_folder+'\\UPP.x_p_'+str(file_seq)+'.txt',np.reshape(x_p.to_numpy(),(n_particle,dim)),delimiter=',' )
    #print('UPP.x_p:',x_p)

def releaseSubStep():
    reset(m_g,v_g,f_g)
    P2G(mP,x_p,v_p,m_g,v_g)
    #computeVolumeNDensity(mP,x_p,V_p,m_g)
    computeGridForce(x_p,F_E_p,F_P_p,f_g)
    updateGridVelocity(m_g,v_next_g,v_g,f_g)
    collideGridBody(m_g,v_next_g)
    #solve linear system: explict|semi-implicit
    updateDeformationGradient(x_p,F_E_p,F_P_p,v_next_g)
    updateParticleVelocity(x_p,v_p,v_next_g,v_g)
    collideParticleBody(x_p,v_p)
    updateParticlePosition(x_p,v_p)

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

init(x_p,v_p,F_E_p,F_P_p)
gui = ti.GUI("Taichi MPM-2013", res=512, background_color=0x112F41)
colors = np.array([0x068587,0xEEEEF0, 0xED553B], dtype=np.uint32)
firstStep()
for frame in range(101):
    print(frame)
    #print(m_g.to_numpy())
    pos=x_p.to_numpy()
    if export_file:
        writer = ti.PLYWriter(num_vertices=n_particle)
        writer.add_vertex_pos(pos[:, 0], pos[:, 1], pos[:, 2])
        writer.export_frame(frame, export_file)
    gui.circles(T(pos), radius=1.5, color=colors[np.ones(n_particle,int)])
    gui.show() # Change to gui.show(f'{frame:06d}.png') to write images to disk
    #
    for t in range(time_steps):
        subStep()
'''
for frame in range(2,5):
    file_seq=frame
    print(frame)
    #print(m_g.to_numpy())
    pos=x_p.to_numpy()
    if export_file:
        writer = ti.PLYWriter(num_vertices=n_particle)
        writer.add_vertex_pos(pos[:, 0], pos[:, 1], pos[:, 2])
        writer.export_frame(frame, export_file)
    gui.circles(T(pos), radius=1.5, color=colors[np.ones(n_particle,int)])
    gui.show() # Change to gui.show(f'{frame:06d}.png') to write images to disk
    releaseSubStep()
    #for t in range(int(4e-3/dt)):
    #debugSubStep()
    #np.savetxt(export_folder+'\\P2G.v_p_'+str(file_seq)+'.txt',np.reshape(v_p.to_numpy(),(n_particle,dim)),delimiter=',' )
    #np.savetxt(export_folder+'\\P2G.v_g_'+str(file_seq)+'.txt',np.reshape(v_g.to_numpy(),(a_grid,dim)),delimiter=',' )
    #np.savetxt(export_folder+'\\CGF.f_g_'+str(file_seq)+'.txt',np.reshape(f_g.to_numpy(),(a_grid,dim)),delimiter=',' )
    #np.savetxt(export_folder+'\\UGV.v_next_g_'+str(file_seq)+'.txt',np.reshape(v_next_g.to_numpy(),(a_grid,dim)),delimiter=',' )
    #np.savetxt(export_folder+'\\CGB.v_next_g_'+str(file_seq)+'.txt',np.reshape(v_next_g.to_numpy(),(a_grid,dim)),delimiter=',' )
    #np.savetxt(export_folder+'\\UDG.F_E_p_'+str(file_seq)+'.txt',np.reshape(F_E_p.to_numpy(),(n_particle*dim,dim)),delimiter=',' )
    #np.savetxt(export_folder+'\\UDG.F_P_p_'+str(file_seq)+'.txt',np.reshape(F_P_p.to_numpy(),(n_particle*dim,dim)),delimiter=',' )
    #np.savetxt(export_folder+'\\UPV.v_p_'+str(file_seq)+'.txt',np.reshape(v_p.to_numpy(),(n_particle,dim)),delimiter=',' )
    #np.savetxt(export_folder+'\\CPB.v_p_'+str(file_seq)+'.txt',np.reshape(v_p.to_numpy(),(n_particle,dim)),delimiter=',' )
    #np.savetxt(export_folder+'\\UPP.x_p_'+str(file_seq)+'.txt',np.reshape(x_p.to_numpy(),(n_particle,dim)),delimiter=',' )
'''
