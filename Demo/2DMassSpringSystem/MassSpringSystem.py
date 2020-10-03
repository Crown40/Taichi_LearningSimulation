import taichi as ti
from ParticleSystem import ParticleSystem
from abc import abstractmethod
from functools import reduce
from Integrator import *
import numpy as np
import copy

ti.init(arch=ti.cpu)

@ti.data_oriented
class MassSpringSystem(ParticleSystem):
    def __init__(self,position,unitAmount_particle,order:int,elemtype:ti.template=ti.Vector,dimension:list=[None],datatype=ti.f32,offset=None,needs_grad=False,is_implict=False):
        self.unitAmount_particle=unitAmount_particle
        shape=[unitAmount_particle*unitAmount_particle]
        ParticleSystem.__init__(self,order,elemtype,dimension,datatype=ti.f32,shape=shape,offset=None,needs_grad=False)
        self.position=position
        self.interval=1/unitAmount_particle
        self.total_force=ti.Vector(dimension[0],datatype,())
        self.spring_force=ti.Vector(dimension[0],datatype,())
        self.damping_force=ti.Vector(dimension[0],datatype,())
        self.mass=ti.var(datatype,())
        self.spring_stiffness=ti.var(datatype,())
        self.rest_length=ti.var(datatype,())
        self.damping=ti.var(datatype,())
        self.wind_force=ti.Vector(dimension[0],datatype,())
        self.init_vtr=[];self.init_mat=[]
        if(is_implict==True):
            self.mat=ti.Matrix(dimension[0],dimension[0],datatype,(),offset,needs_grad=needs_grad)
            self.identity_mat=ti.Matrix(dimension[0],dimension[0],datatype,(),offset,needs_grad=needs_grad)
            #self.dfDivdx_mat=ti.Matrix(dimension[0],dimension[0],datatype,shape,offset,needs_grad=needs_grad)
            #self.dfDivdv_mat=ti.Matrix(dimension[0],dimension[0],datatype,shape,offset,needs_grad=needs_grad)
        #self.stepSize[None]=ti.sqrt(self.mass/self.spring_stiffness)
        #self.init()

    '''
    '''
    @ti.kernel
    def init(self):# []:get_item in Py-Scope
        self.mass[None]=1000
        self.spring_stiffness[None]=0.1
        self.rest_length[None]=self.interval*2
        self.damping[None]=0.01
        self.wind_force[None]=0.1*ti.Vector([1.0,0.0])
        if(self.is_implict==True):
            for H in ti.grouped(self.identity_mat):
                for i in ti.static(range(self.identity_mat.n)):
                    self.identity_mat[H][i,i]=1.0
        #
        for i in ti.static(range(self.dimension[0])):
            self.init_vtr.append(0.0)
            self.init_mat.append([])
            for j in ti.static(range(self.dimension[0])):
                self.init_mat[i].append(0.0)
        #
        inx=0
        n=self.dimension
        for i in range(self.unitAmount_particle):
            for j in range(self.unitAmount_particle):
                self.state[0][inx]=self.position+self.interval*ti.Vector([j*1.0,i*1.0])
                inx+=1
    '''
    '''
    def setParameter(self,unitAmount_particle=None,mass=None,spring_stiffness=None,rest_length=None,damping=None,stepSize=None):
        if(unitAmount_particle!=None):
            self.unitAmount_particle=unitAmount_particle
        if(mass!=None):
            self.mass=mass
        if(spring_stiffness!=None):
            self.spring_stiffness=spring_stiffness
        if(rest_length!=None):
            self.rest_length=rest_length
        if(damping!=None):
            self.damping=damping
        if(stepSize!=None):
            self.stepSize=stepSize
    '''
    '''
    @ti.pyfunc
    def indexOf(self,i,j):
        return i*self.unitAmount_particle+j

    # Shiftness N Damping
    @ti.func
    def springAnalysis(self,inx,i,j):
        relative_inx=self.indexOf(i,j)
        x_ab=self.state[0][inx]-self.state[0][relative_inx]
        normalize_xab=x_ab.normalized();norm_xab=x_ab.norm()
        self.spring_force+=(norm_xab-self.rest_length)*normalize_xab
        v_ab=self.state[1][inx]-self.state[1][relative_inx]
        self.damping_force+=normalize_xab*v_ab*normalize_xab
        if(self.is_implict==True):
            outer_product_xab=normalize_xab.outer_product(normalize_xab)
            self.mat+=(1.0-self.rest_length/norm_xab)*(self.identity_mat-outer_product_xab)+outer_product_xab
    '''
    '''
    # Force Analysis
    @ti.kernel
    def evalf(self):
        bound=self.unitAmount_particle-1
        lbound_flex=1;hbound_flex=self.unitAmount_particle-2
        reci_mass=1/self.mass
        nega_spring_stiffness=-self.spring_stiffness
        nega_damping=-self.damping
        

        for inx in range(self.unitAmount_particle,reduce(lambda x,y: x*y,self.shape)):
            if(self.is_implict==True):
                self.mat=ti.Matrix(self.init_mat)
            
            self.total_force=ti.Vector(self.init_vtr)
            self.spring_force=ti.Vector(self.init_vtr)
            self.damping_force=ti.Vector(self.init_vtr)
            i=inx//self.unitAmount_particle;j=inx%self.unitAmount_particle
                
            if(i>0):#StructSpring
                self.springAnalysis(inx,i-1,j)
                if(j>0):#ShearSpring
                    self.springAnalysis(inx,i-1,j-1)
            #
            if(i<bound):
                self.springAnalysis(inx,i+1,j)
                if(j<bound):
                    self.springAnalysis(inx,i+1,j+1)
            #
            if(j>0):
                self.springAnalysis(inx,i,j-1)
                if(i<bound):
                    self.springAnalysis(inx,i+1,j-1)
            #
            if(j<bound):
                self.springAnalysis(inx,i,j+1)
                if(i>0):
                    self.springAnalysis(inx,i-1,j+1)
            #Flex|Bend Spring
            if(i>lbound_flex):
                self.springAnalysis(inx,i-2,j)
            if(i<hbound_flex):
                self.springAnalysis(inx,i+2,j)
            if(j>lbound_flex):
                self.springAnalysis(inx,i,j-2)
            if(j<hbound_flex):
                self.springAnalysis(inx,i,j+2)
            #
            self.spring_force*=nega_spring_stiffness
            self.damping_force*=nega_damping
            self.total_force+=self.spring_force
            self.total_force+=self.damping_force
            if(j==0):
                self.total_force+=self.wind_force
            #self.state[1][inx]*=ti.exp(-self.stepSize*self.damping)
            self.state[2][inx]=reci_mass*self.total_force
            if(self.is_implict==True):
                self.dfDivdx_mat[inx]=self.mat*nega_spring_stiffness
                for k in ti.static(range((self.mat.n))):
                    self.dfDivdx_mat[inx][k,k]*=reci_mass
    '''
    '''

num=4
lines=[]
for i in range(2):
    lines.append(ti.Vector(2,ti.f32,shape=(2*num*(num-1)+(num-1)*(num-1)*2)))
system=MassSpringSystem(ti.Vector([0.1,0.1]),num,order=2,elemtype=ti.Vector,dimension=[2],is_implict=True)
specie=input('请输入积分方法(Plz input the num of Integration Method): 1:ForwardEuler 2:SemiImplicitEuler 3:RK4 4:BackwardEuler\n:')
if(specie==1):
    integrator=ForwardEuler(stepSize=0.1)
elif(specie ==2):
    integrator=SemiImplicitEuler(stepSize=0.1)
elif(specie==3):
    integrator=RK4(system=system,stepSize=0.1)
else:
    integrator=BackwardEuler(system=system,stepSize=0.1)

h=800;w=800;
gui=ti.GUI("MassSpringSystem",(w,h),0xffffff)


'''
'''
@ti.kernel
def paintLine():
    sys=system
    bound=sys.unitAmount_particle-1
    cnt=0
    for i in range(sys.unitAmount_particle):
        for j in range(sys.unitAmount_particle):
            if (i<bound):
                lines[0][cnt]=sys.state[0][sys.indexOf(i,j)];lines[1][cnt]=sys.state[0][sys.indexOf(i+1,j)]
                cnt+=1
                if(j<bound):
                    lines[0][cnt]=sys.state[0][sys.indexOf(i,j)];lines[1][cnt]=sys.state[0][sys.indexOf(i+1,j+1)]
                    cnt+=1
            if(j<bound):
                lines[0][cnt]=sys.state[0][sys.indexOf(i,j)];lines[1][cnt]=sys.state[0][sys.indexOf(i,j+1)]
                cnt+=1
                if(i>0):
                    lines[0][cnt]=sys.state[0][sys.indexOf(i,j)];lines[1][cnt]=sys.state[0][sys.indexOf(i-1,j+1)]
                    cnt+=1
                    #gui.line(sys.state[0][sys.indexOf(i,j)], sys.state[0][sys.indexOf(i-1,j+1)], color = 0xff0000, radius = 5)

        #
        #gui.lines(lines[0],lines[1],color = 0xff0000,radius=5)

'''
'''
def func():
    system.init()
    system.setParameter()
    #backup_integrator.subStep(system)
    while True:
        integrator.subStep(system)
        paintLine()
        gui.lines(lines[0].to_numpy(),lines[1].to_numpy(),color = 0xff0000,radius=5)
        gui.circles(system.state[0].to_numpy(),0x0000ff,radius=10)
        #gui.circle([0.5,0.5], color = 0xff0000, radius = 100)
        gui.show()
        #ti.video()
        
func()
