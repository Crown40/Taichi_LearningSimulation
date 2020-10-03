import taichi as ti
from abc import abstractmethod
from ParticleSystem import ParticleSystem
import copy
from functools import reduce

#ti.init(arch=ti.gpu)

# No var
def generateSameScaleTensor(elemtype:ti.template,dimension:list,datatype,shape=[],offset=None,needs_grad=False):
    ins=None
    if (elemtype==ti.Matrix):
        ins=ti.Matrix(dimension[0],dimension[1],datatype,shape,offset,needs_grad=needs_grad)
    else:
        ins=ti.Vector(dimension[0],datatype,shape,offset,needs_grad=needs_grad)
    return ins

class Integrator:
    # ExplictApproach
    # FORWARD_EULER="ForwardEuler";SEMIIMPLICIT_EULER="SemiImplicitEuler"
    # RK4="RK4"
    # ImplictApproach:UnPerpared
    # MIDDLE_POINT="MiddlePoint";BACKWARD_EULER="BackwardEuler"
    def __init__(self,stepSize=0.1):
        self.stepSize=stepSize

    def setParameter(stepSize=None):
        if(stepSize!=None):
            self.stepSize=stepSize
    
    @abstractmethod
    def subStep(system):
        pass
    @abstractmethod
    def takeStep():
        pass
   
# Explicit
@ti.data_oriented
class ForwardEuler(Integrator):
    def __init__(self,stepSize=0.1):
        self.stepSize=stepSize

    def subStep(self,system:ParticleSystem):
        system.evalf()
        self.takeStep(self.stepSize,system.state[0],system.state[1],system.state[2])
    @ti.kernel
    def takeStep(self,stepSize:ti.template(),state:ti.template(),first_derivative:ti.template(),second_derivative:ti.template()):
        #stepSize=self.stepSize
        for I in ti.grouped(state):
            state[I]+=stepSize*first_derivative[I]
            first_derivative[I]+=stepSize*second_derivative[I]

@ti.data_oriented
class SemiImplicitEuler(Integrator):
    def __init__(self,stepSize=0.1):
        self.stepSize=stepSize

    def subStep(self,system:ParticleSystem):
        system.evalf()
        self.takeStep(self.stepSize,system.state[0],system.state[1],system.state[2])
    @ti.kernel
    def takeStep(self,stepSize:ti.template(),state:ti.template(),first_derivative:ti.template(),second_derivative:ti.template()):
        for I in ti.grouped(state):
            first_derivative[I]+=stepSize*second_derivative[I]
            state[I]+=stepSize*first_derivative[I]

@ti.data_oriented
class RK4(Integrator):
    def __init__(self,system:ParticleSystem,stepSize=0.1):
        self.stepSize=stepSize
        #self.inx_state_range=ti.Vector([0,1])
        #self.inx_full_range=ti.Vector([0,system.order+1])
        self.k=[]
        for i in range(4):
            self.k.append([])
            for j in range(system.order+1):
                self.k[i].append(generateSameScaleTensor(system.elemtype,system.dimension,system.datatype,system.shape,system.offset,system.needs_grad))

    def subStep(self,system:ParticleSystem):
        k=self.k
        order=system.order
        for i in range(4):
            for j in range(0,1):
                self.cover(k[i][j],system.state[j])
            if(i==0):# y_0->k1=f(y_0)
                system.evalf()
            elif(i==1 or i==2):# y_1=y_0+h/2*k1->k2=f(y_1)||# y_2=y_0+h/2*k2->k3=f(y_2)
                for j in range(0,1):
                    self.cover(system.state[j],k[0][j])
                self.takeStep(self.stepSize/2,system.state[0],system.state[1],system.state[2])
                system.evalf()
            else:# y_3=y_0+h/2*k3->k4=f(y_3)
                for j in range(0,1):
                    self.cover(system.state[j],k[0][j])
                self.takeStep(self.stepSize,system.state[0],system.state[1],system.state[2])
                system.evalf()
        # y_4=y_0+h/6*k1+h/3*k2+h/3*k3+h/6*k4
        for j in range(system.order,system.order+1):
            self.sumDerivative(self.stepSize,k[1][j],k[2][j],k[3][j],system.state[j])
        for j in range(0,1):
            self.cover(system.state[j],k[0][j])
        self.takeStep(self.stepSize,system.state[0],system.state[1],system.state[2])

    @ti.kernel
    def cover(self,state0:ti.template(),state1:ti.template()):
        for I in ti.grouped(state0):
            state0[I]=state1[I]

    @ti.kernel
    def sumDerivative(self,stepSize:ti.template(),k1:ti.template(),k2:ti.template(),k3:ti.template(),k4:ti.template()):
        coef1=stepSize/6;coef2=stepSize/3
        #for i in range(irange[0],irange[irange.n-1]):
            #state1=k1[i];state2=k2[i];state3=k3[i];state4=k4[i]
        for I in ti.grouped(k4):
            k4[I]+=k1[I];k4[I]*=coef1
            k3[I]+=k2[I];k3[I]*=coef2
            k4[I]+=k3[I]
    '''
    '''
    @ti.kernel
    def takeStep(self,stepSize:ti.template(),state:ti.template(),first_derivative:ti.template(),second_derivative:ti.template()):
        for I in ti.grouped(state):
            first_derivative[I]+=stepSize*second_derivative[I]
            state[I]+=stepSize*first_derivative[I]

# Impilcit
# Please use Vector OR Matrix as Elemtype:Faster
@ti.data_oriented
class BackwardEuler(Integrator):
    # Solution:Jacobi_Iteration
    def __init__(self,system:ParticleSystem,stepSize=0.1):
        elemtype=system.elemtype
        dimension=system.dimension
        datatype=system.datatype
        shape=system.shape
        offset=system.offset
        needs_grad=system.needs_grad
        order=system.order
        self.stepSize=stepSize
        #self.diff_mat=ti.Matrix(dimension[len(dimension)-1],dimension[0],datatype,shape)
        #self.hOrder_state=generateSameScaleTensor(elemtype,dimension,datatype,shape,offset,needs_grad)
        self.b=generateSameScaleTensor(elemtype,dimension,datatype,shape,offset,needs_grad)
        self.v=generateSameScaleTensor(elemtype,dimension,datatype,shape,offset,needs_grad)
        #self.lOrder_state=generateSameScaleTensor(elemtype,dimension,datatype,shape,offset,needs_grad)
    '''
    '''
    def subStep(self,system):# Computing (mass_matrix)^-1 have been embeded in system.state[system.order]
        system.evalf()
        self.preprocess(system.state[1],system.state[system.order],system.dfDivdx_mat)
        self.iteration_Jacobi(system.state[1],system.dfDivdx_mat)
        self.takeStep(self.stepSize,system.state[0],system.state[1])
    '''
    '''
    @ti.kernel
    def preprocess(self,first_derivative:ti.template(),second_derivative:ti.template(),A:ti.template()):
        stepSize=self.stepSize
        for I in ti.grouped(self.b):
            #self.hOrder_state[I]=second_derivative[I]
            self.b[I]=first_derivative[I]
            self.v[I]=first_derivative[I]
            A[I]*=stepSize*stepSize
            #self.lOrder_state[I]=state[I]
        # A:Matrix
        # b:Vector
        for I in ti.grouped(self.b):
            self.b[I]+=stepSize*second_derivative[I]
            for j in ti.static(range(A.n)):
                A[I][j,j]-=1.0
                A[I][j,j]=-A[I][j,j]
    '''
    '''
    @ti.kernel
    def iteration_Jacobi(self,first_derivative:ti.template(),A:ti.template()):
        for H in ti.grouped(self.b):
            for i in ti.static(range(self.b.n)):
                r=self.b[H][i]
                r+=A[H][i,i]*self.v[H][i]
                for j in ti.static(range(self.b.n)):
                    r-=A[H][i,j]*self.v[H][j]
                #tag#
                first_derivative[H][i]=r/A[H][i,i]
    '''
    '''
    @ti.kernel
    def takeStep(self,stepSize:ti.template(),state:ti.template(),first_derivative:ti.template()):
        for I in ti.grouped(state):
            state[I]+=stepSize*first_derivative[I]





