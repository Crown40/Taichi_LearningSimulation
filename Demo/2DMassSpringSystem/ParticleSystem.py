import taichi as ti
from abc import abstractmethod
from functools import reduce

#ti.init(arch=ti.cpu)

# Physical Particle System
class ParticleSystem:
    # VAR=0;VECTOR=1;MATRIX=2;
    # ElemType:ti.var:dimension=[1],Be Specified as specific value:(value) for HighD=>1D
    def __init__(self,order:int,elemtype:ti.template=ti.Vector,dimension:list=[None],datatype=ti.f32,shape:list=[],offset=None,needs_grad=False,stepSize=0.1,is_implict=False):
        self.elemtype=elemtype
        self.dimension=dimension
        self.datatype=datatype
        self.shape=shape
        self.offset=offset
        self.needs_grad=needs_grad
        self.is_implict=is_implict
        if(is_implict==True):
            self.dfDivdx_mat=ti.Matrix(dimension[0],dimension[0],datatype,shape,offset,needs_grad=needs_grad)
        self.state=[]
        # variable N its derivative: 0:variable 1:1st_derivative ... i:ith_derivative
        self.order=order
        if(elemtype==ti.var):
            for i in range(order+1):
                self.state.append(ti.var(dt=datatype,shape=shape,offset=offset,needs_grad=needs_grad))
        elif(elemtype==ti.Matrix):
            # row x col
            for i in range(order+1):
                self.state.append(ti.Matrix(n=dimension[0],m=dimension[1],dt=datatype,shape=shape,offset=offset,needs_grad=needs_grad))
        else:
            for i in range(order+1):
                self.state.append(ti.Vector(dimension[0],dt=datatype,shape=shape,offset=offset,needs_grad=needs_grad))

    # Initialization
    @abstractmethod
    def init():
        pass
    # Integrated function
    @abstractmethod
    def evalf():
        pass
    # Alter Parameter
    @abstractmethod
    def setParameter(stepSize):
        pass

