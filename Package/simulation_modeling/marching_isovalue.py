import taichi as ti
import numpy as np
import base64
import math
 

vertex_statuses=np.array([1,8,16,128,2,4,32,64]) #np.array([1,2,4,8,16,32,64,128]) # 
_et2 = np.array([
        [[-1, -1], [-1, -1]],  #none
        [[0, 1], [-1, -1]],  #a
        [[0, 2], [-1, -1]],  #b
        [[1, 2], [-1, -1]],  #ab
        [[1, 3], [-1, -1]],  #c
        [[0, 3], [-1, -1]],  #ca
        [[2, 3], [0, 1]],  #cb
        [[2, 3], [-1, -1]],  #cab
        [[2, 3], [-1, -1]],  #d
        [[2, 3], [0, 1]],  #da
        [[0, 3], [-1, -1]],  #db
        [[1, 3], [-1, -1]],  #dab
        [[1, 2], [-1, -1]],  #dc
        [[0, 2], [-1, -1]],  #dca
        [[0, 1], [-1, -1]],  #dcb
        [[-1, -1], [-1, -1]],  #dcab
    ], np.int32)

_et3b = b'|NsC0|NsC0|NsC0|Ns902m}BB|NsC0|NsC0|Nj613IG59|NsC0|NsC0{{aXC2?zoI|NsC0|NsC00RjsD|NsC0|NsC0|Ns902m=8E3jhEA|NsC0|NjXB3IGBL|NsC0|NsC0{{jdD0tyHU2?+oH|NsC00}BHG|NsC0|NsC0|Ns903jzoW0RR90|NsC0|Nj9A00ILG|NsC0|NsC0{{agE0SOBU2n+xJ|NsC00}25P3IqTD|NsC0|Ns903IPBJ3J41d|NsC0|NjFC00RpN3knJU|NsC0{|N{R3J44T|NsC0|NsC01P2KJ|NsC0|NsC0|Ns940{{mD1poj4|NsC0|Nj612?zuS|NsC0|NsC0{{#UE1P1{J0|Ed4|NsC00RjpL1PA~B|NsC0|Ns931P22E1OWmH|NsC0|NjXB3JCxL2m}ZJ|NsC0{{jjL0tp8K2LlHQ1poj42m}WM3j+WD|NsC0|Ns9B1P2QO1OfmA|NsC0|NjX90SE*K0s{;G|NsC0{{#mM2?PrX3jzrO0ssI10}25H3knAa1poj4|Ns913km@Q3jqKG2MYxM|NjIB2nhfS2@47f00aO3{{#mM1PciX3kv`L|NsC02?YfI|NsC0|NsC0|Ns991q1*H1ONa3|NsC0|Nj651OWvA|NsC0|NsC0{|E&H2m=KJ0R{j6|NsC00RjpM1qA>9|NsC0|Ns9300;pB3IquS|NsC0|NjL73IzlL1ONj6|NsC0{{jjH0|EsD1q1^G2><{82?YcK0}KEE|NsC0|Ns903jzQL3j_%T|NsC0|Nj651ONdA0s{;G|NsC0{{jI80tE;H2nz%V1^@s53IhuY0RssI1poj4|Ns942?YQM0SF2K2n!1T|NjL900jUG1q%ub00aO3{{;jH1qccX2n+xJ|NsC02?q!T2MPcG|NsC0|Ns990{{sH0|f^I|NsC0|Nj672mk>G0R;#D|NsC0{{aO90|f{F|NsC0|NsC02?q!X1qTWN0{{R2|Ns9A0RjmH00jd81qTEF|NjU80tf;H2n7cU1p@#7{{jjH0tEvD1qc8C|NsC02MGlS2nhoV0{{R2|Ns991qTTS0to^D0tXBK|NjC53jhHK0S5>H1qc8C{|f>E3jqfQ0R{j6|NsC02?YoU1qTWN0}2BR|Ns952LJ^C2?q-R0RRdM3IGcV01E&E3IzZN00#vJ01FBQ2MY!N|NsC0|NsC03I+xL|NsC0|NsC0|Ns902m=KQ2LJ#6|NsC0|NjX90R;*M|NsC0|NsC0{{aXC0SO2N3I_lG|NsC00R{yE1_A&7|NsC0|Ns911_c2E1_J;H|NsC0|NjXF1qlEK00IX8|NsC0{{;yM1qcEK0tN#D2><{80s{*Q1_l5B|NsC0|Ns9B00;{L015^L|NsC0|Nj612?7HP1qufL|NsC0{{;#L0SN*L3jzrU3;+NB1_KKQ1p@^E1ONa3|Ns902nzrU1poyB1q%lM|NjFE1^@#F00spB1quKE{{{sK1_=ub2?+oH|NsC01qucP2MGWF|NsC0|Ns940{{dE0|o^O|NsC0|Nj9A00jyL2m}ZJ|NsC0{|W{L0SN~I2LlHQ1poj41_1&F1px#H2><{8|Ns910tE#E1_J;D0|W>E|NjUC2MGWL00spB0tWy8{|5sJ2MGiN0tp2P1_A~N0}BEN2m}fS1^@s5|Ns953I+rR0t5m80tXBK|Nj612?PfS0s{*L3I_lG{|N#C2@3)V1PccX1O*BP2m}WM3k3rO0R;;N|Ns950Sg5S1_1yI2MYuM1PcHK2>=EK00RaK1_KBL2L=TR1_=uU2MGrY3IG593IquT1PcHE|NsC0|Ns943I+rT3IGTL|NsC0|NjaA0SX2H1_S{A|NsC0{|EyC2muBN1_TBH3jhEA0R#yF0t5mE1poj4|Ns9300;pB2?7KO0tN*C|Nj621Ox&G|NsC0|NsC0{|EyD2m%BI0tWy8|NsC03IquX1_TQN1ONa3|Ns902m%5K3j_%Y1PTWK|NjFE0ssL900smG0Sf>B{{{pB1_25L2mt~C3kVAV2?hiS0|p5J0}BQN|Ns983jqiL01E~I2>}EK1OWpJ1_K5F00spA|NsC0{{{pI3kC@P|NsC0|NsC02MPuU2nq-Z3jhEA|Ns902Lk{K2LK5Q1_uiN|NjaG2LTEP0S5>H2mt^8{|W{N3I_oJ2Lu2A|NsC00RjdA1_%KN2?z!U|Ns921_=TQ0R{&N00{#J0|^HR00#gD1^@#8|NsC0{|5sC1_uKF|NsC0|NsC00s{*Q1_%lW2?z!U|Ns9200#mG3jhfR1_ufW3I_oQ009RG0SX5O2MPiM3kw1P3jqfT1_1^K0ssI12nhxV1_ucN1`7rQ0RsjA2>}ZR2mk;7|NsC0|NjRF00#gD0}B8P1_1y6{|5^O|NsC0|NsC0|NsC02L=oO|NsC0|NsC0|Ns9300;{Q2LJ#6|NsC0|Nj612@3}X|NsC0|NsC0{|EsI2m=8N2L}KD|NsC03IPHJ3kU!I|NsC0|Ns910ty2F2nGuW|NsC0|NjCB00IgL1`7xO|NsC0{{{;O0ty2P2m=ZU2><{82Lb~I0tf&9|NsC0|Ns9700;*L00sg8|NsC0|NjC91_A>I009aA|NsC0{{aR90SE>G2?z)W2LJ#63I_%X0S5sC2mk;7|Ns9A2L=HL3IPZQ0RRa9|Nj632LJ~O0162P3J3rH{|5#N2MP!X3JL%J|NsC01_%TT2nPTE|NsC0|Ns931`7iK1^@&G|NsC0|NjUE3kU=T2>=2A|NsC0{|N*J2?hfR0|5&I2LJ#61_%TO3kU)V0ssI1|Ns910ty2F3jhWS00ajA|NjIF2m}TT00IdD3JL%J{|X5M3IhTO1Op2L1_TBJ2m%8L1OfyG0{{R2|Ns901OfyG0{{R2|NsC0|Nj9A00IL90t5yG0|@{B{{aaE0R#d91P1^A|NsC02mu2K1_1~J1_lZN|Ns9A0RRdB1_l5G|NsC0|NjIA0|WyI1_}cJ0|^QV0}2TQ1_}iK|NsC0|NsC01PKKP1`GfH|NsC0|Ns902m=HO1q%lT|NsC0|NjL50R;pA2L=oO|NsC0{|g5O2m=HI1q1^D1^@s52?YcS0RjgG3;+NB|Ns963kLxL3IGTL1PKNI|NjRD3k3uU1Of^K00RI2{{sXF0|f*F0tE^M0t*KQ2Lb~J1_A{H3IG59|Ns991q1*H1^@;E1_%fL|NjF90s{vI0R;dB1OWg4{{{jG1_%cN0SE*L1px&J2?YcS0R{mF1_1*H|Ns911_}WO1_1yE2nPTO1q1{D3IqxT00RmJ3I_uR3I_%X2MP!U1PTNQ3jhEA1_=cQ3keGd3IG59|Ns931`7ZN0{{gE00{;E|Nj6B2ml2O009LB1`GfH{{{;K1_K2J0|Ed4|NsC00RjpM1q%ra2nz)U|Ns903j+WK3jhfQ1qKNL0tyQV1q%fR2ml2N1p)v91qKTP1_K2G3IhrS1ONa31qcZR0tf{L0s{yF|Ns991qKNQ000I8|NsC0|Nj962muHH1qKKM2m%HI2mu8K0s#j9|NsC0|NsC00RsjB1_}cR1_cHQ2nhxX0RRdB1_=cK1qJ~B|Nj632n7ZT|NsC0|NsC0{|W^L|NsC0|NsC0|NsC03k3=X1q=WG|NsC0|Ns9B1qurX1qcHG|NsC0|NjLG2L%cX0SN&A|NsC0{|W~M3JV7b2muHK0ssI13jqQP2LT5K0ssI1|Ns902m=8E2LT5K2LcQK|NjXG1qlKN2>=2D3kU!I{|5yE2LcNP2?7HG2nh%R0tE^J0|f&I1^@s5|Ns980ssgF0tg2M3IYZH|NjX90R;*J1p@~I3IhNC{|N{J2?7BK2LcKL1qTHJ0RsgC2L=EC|NsC0|Ns902nPTM0RaaE|NsC0|NjX90|^5K1p^2F|NsC0{|N{O1qlcL|NsC0|NsC01qcKM3J3}d2><{8|Ns9500adK00jyQ3j+ZE|Nj612?zuV2nq`d1O@;9{|XBP3IqiU0|W^K0RsU90tEpA2n7NQ2m}QP|Ns901PcHQ0|W&N0t*2J0Sf>E1poyJ0t*EM1qcfW1qlQN0t*BG|NsC0|NsC00tE^K1p)&E1p^2K|Ns953IYWJ1Ox&A|NsC0|NjFD0s{pK0|*5K1qc8E2?YuQ1p)*C2?7ZO0{{R22m}QP1p@;G0ssI1|Ns901O)*A1^@s5|NsC0|NjUC1qcNL2>=BE0|o#8{|N*I|NsC0|NsC0|NsC01PccQ2@44d3;+NB|Ns902m=HO2MG%Y2?`7U|Nj9B3jqrR0R#XC1PlNF{{sO80|W>G3IqoP3knMa1PccV3j_%Q3kd-N|Ns992LuTV2MGZS0t*2E2m=cT1PcTL0t5j6|NsC0{|g5M3j_iP0|WyC1poj40tpHN2MGcL2L}WR|Ns993I_=X1PTHN2nPTH00#pH3IhrP2LuWM3IGHE3IPfN2nPiJ|NsC0|NsC01PK8I0S5;G1ONa3|Ns942>}EF2LK2G2nPZG|NjI40|x{H|NsC0|NsC0{{#pJ|NsC0|NsC0|NsC02?_`b3kd)J|NsC0|Ns9300{#L3kwMf|NsC0|Nj613IGZS2nq}T|NsC0{{sOE3j+%O|NsC0|NsC00RjsF3keAe2><{8|Ns9300{#L3jqQN0t*TM|Nj623kU!U|NsC0|NsC0{{sRG|NsC0|NsC0|NsC00s{yF2nq@a3IG59|Ns993IYHL0{{R2|NsC0|NjC52m%NS009UA3JCxI{{adD|NsC0|NsC0|NsC00RspL0SN#9|NsC0|Ns902?78A|NsC0|NsC0|Nj632><{8|NsC0|NsC0|NsC0|NsC0|NsC0|NsC0'
_et3 = np.frombuffer(base64.b85decode(_et3b), dtype=np.int8).reshape(256, 5, 3).astype(np.int32)

PIf=3.1415926

@ti.data_oriented
class McIsovalue:
    def __init__(self, dim = 3, res = 512, space_sideLength = 1.0, has_normal=False, is_smooth=False):
        # side_length
        #self.weight=weight
        self.unit_sideLength=space_sideLength*1/res
        self.inv_unitSideLength=1/self.unit_sideLength
        #self.radius=radius if(radius!=None) else self.unit_sideLength 
        occur_range=1 #int(math.ceil(self.radius/self.unit_sideLength)) if(radius!=None) else 1
        #self.occur_range=occur_range if(radius!=None) else 1 
        #self.occur_ndrange=[[-occur_range,occur_range+1] for _ in range(dim)] 
        self.neighbor_ndrange=[[-1,2] for _ in range(dim)]
        self.cube_ndrange=[[0,2] for _ in range(dim)]
        # layout
        # space,area,page,block,grid,cell,leaf,unit,individual 
        self.dim=dim
        self.res=res
        sqrt_res=int(math.sqrt(res))
        indices=[ti.ij,ti.ijk][dim-2]
        pId_index=[ti.k,ti.l][dim-2]
        self.grid_size=occur_range*(2**dim) #2**dim
        self.grid_size=sqrt_res if(self.grid_size>sqrt_res) else self.grid_size 
        self.space_size=int(res//self.grid_size)
        self.cell_size=[8,16][dim-2] #(self.grid_size)**dim
        chunk_size=4
        # boundary
        self.gp_maxAmount=ti.field(int,())
        self.g_pID=ti.field(int)
        self.space=ti.root.pointer(indices,self.space_size)
        self.grid=self.space.pointer(indices,self.grid_size) # dense|bitmasked
        self.g_pIDs=self.grid.dynamic(pId_index,self.cell_size,chunk_size)
        self.g_pIDs.place(self.g_pID)
        self.g_pAmount=ti.field(int)
        self.grid.place(self.g_pAmount)
        # cube
        self.cube_threshold=1.0
        self.cube_index=ti.field(int)
        self.cube_isovalue=ti.field(float)
        self.grid.place(self.cube_index)
        self.grid.place(self.cube_isovalue)
        if(has_normal):
            self.cube_grad=ti.Vector.field(dim,float)
            self.grid.place(self.cube_grad)
        #   cube_vertex_table
        self.cube_table=ti.field(int,vertex_statuses.shape[0])
        @ti.materialize_callback
        def init_cubetables():
            self.cube_table.from_numpy(vertex_statuses)
        #   edge_table
        et=[_et2,_et3][dim-2]
        self.edge_table=ti.Vector.field(dim,int,et.shape[:2])
        @ti.materialize_callback
        def init_edgetables():
            self.edge_table.from_numpy(et)
        # face
        self.faces_amount=ti.field(int,())
        self.vertices_amount=ti.field(int,())
        self.vertices_datasize=2*[4,24][dim-2]*self.res**2
        print('m.vertices_datasize:'+str(self.vertices_datasize))
        self.vertex_pos=ti.Vector.field(dim,float,self.vertices_datasize)
        self.has_normal=has_normal
        self.is_smooth=is_smooth
        if(has_normal):
            self.vertex_normal=ti.Vector.field(dim,float,self.vertices_datasize)
        # particle_pos
        #self.particlePos_amount=ti.field(int,())
        #self.particlePos_datasize=2*res**2
        #print('m.particlesPos_datasize:',self.particlePos_datasize)
        #self.particle_pos=ti.Vector.field(dim,int,self.particlePos_datasize)
        # Gaussian_weight
        self.gaussian_weight=ti.field(float, 128)

    def marching_cube(self,pos,w0,kernel_drang,sigma):
        self.space.deactivate_all()
        self.scatter(pos,w0)
        self.blur_Gaussian(kernel_drang,sigma)
        self.marching(pos)
        self.meshing()
        
    
    @ti.kernel
    def scatter(self,pos:ti.template(),w0:float):
        # scatter
        for p in pos:
            pos_p=pos[p]
            Xp=pos_p*self.inv_unitSideLength
            base=int(Xp-0.5)
            fx=Xp-base
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
            for offset in ti.static(ti.grouped(ti.ndrange(3, 3, 3))):
                dpos = (offset - fx) / self.res
                weight = float(w0)
                for t in ti.static(range(3)):
                    weight *= w[offset[t]][t]
                self.cube_isovalue[base + offset] += weight
        pass

    def blur_Gaussian(self,kernel_drang,sigma):
        kernel_drang=int(math.ceil(kernel_drang))
        if kernel_drang <= 0: return
        self.sub_blur(kernel_drang, self.gaussian_weight, self.cube_grad(0), 0)
        self.sub_blur(kernel_drang, self.cube_grad(0), self.cube_grad(1), 1)
        self.sub_blur(kernel_drang, self.cube_grad(1), self.gaussian_weight, 2)
        pass

    @ti.kernel
    def init_weightGaussian(self,kernel_drange:int,sigma:float):
        sum = -1.0
        for i in range(kernel_drange + 1):
            x = sigma * i / kernel_drange
            y = ti.exp(-x**2)
            self.gaussian_weight[i] = y
            sum += y * 2
        for i in range(kernel_drange + 1):
            self.gaussian_weight[i] /= sum
     
    @ti.kernel
    def sub_blur(self,kernel_drange:int,source:ti.template(),target:ti.template(),axis:int):
        for I in ti.grouped(target):
            target[I] = src[I] * self.gaussian_weight[0]
        for I in ti.grouped(source):
            for i in range(1, kernel_drange + 1):
                dir = ti.Vector.unit(3, axis,int)
                wei = source[I] * self.gaussian_weight[i]
                target[I + i * dir] += wei
                target[I - i * dir] += wei

    @ti.kernel
    def marching(self,pos:ti.template()):
        # alias
        g_pID,cube_index,cube_isovalue=ti.static(self.g_pID,self.cube_index,self.cube_isovalue)
        # clear vertex
        self.vertices_amount[None]=0
        # Cube_Vertex
        for I in ti.grouped(cube_isovalue):
            cubeIndex=0
            if ti.static(self.dim == 2):
                i, j = I
                if cube_isovalue[i, j] > 1: cubeIndex |= 1
                if cube_isovalue[i + 1, j] > 1: cubeIndex |= 2
                if cube_isovalue[i, j + 1] > 1: cubeIndex |= 4
                if cube_isovalue[i + 1, j + 1] > 1: cubeIndex |= 8
            else:
                i, j, k = I
                if cube_isovalue[i, j, k] > 1: cubeIndex |= 1
                if cube_isovalue[i + 1, j, k] > 1: cubeIndex |= 2
                if cube_isovalue[i + 1, j + 1, k] > 1: cubeIndex |= 32
                if cube_isovalue[i, j + 1, k] > 1: cubeIndex |= 16
                if cube_isovalue[i, j, k + 1] > 1: cubeIndex |= 8
                if cube_isovalue[i + 1, j, k + 1] > 1: cubeIndex |= 4
                if cube_isovalue[i + 1, j + 1, k + 1] > 1: cubeIndex |= 64
                if cube_isovalue[i, j + 1, k + 1] > 1: cubeIndex |= 128
            cube_index[I]=cubeIndex

        # Gradient
        if(ti.static(self.has_normal)):
            for I in ti.grouped(cube_isovalue):
                gradient=ti.Vector.zero(float,self.dim)
                for i in ti.static(range(self.dim)):
                    dI=ti.Vector.unit(self.dim,i,int)
                    gradient[i]=(cube_isovalue[I+dI]-cube_isovalue[I-dI])/self.unit_sideLength
                self.cube_grad[I]=gradient.normalized(1e-5)
        pass

    @ti.kernel
    def meshing(self):
        # face:Triangle
        for I in ti.grouped(self.cube_index):
            cubeIndex=self.cube_index[I]
            refer_pos=I*self.unit_sideLength
            for plane_index in ti.static(range(self.edge_table.shape[1])):
                plane_eIndex=self.edge_table[cubeIndex,plane_index]
                if(plane_eIndex[0]!=-1):
                    #
                    vertex_index=ti.atomic_add(self.vertices_amount[None],self.dim)
                    #
                    for point_num in ti.static(range(self.dim)):
                        pos1,pos2=refer_pos,refer_pos
                        vertex1_Index,vertex2_Index=I,I
                        edge_index=plane_eIndex[point_num]
                        if(ti.static(self.dim==3)):
                            # locate
                            if(edge_index>3 and edge_index<8):
                                pos1.y+=self.unit_sideLength
                                vertex1_Index.y+=1
                            if(edge_index==2 or edge_index==6 or edge_index==10 or edge_index==11):
                                pos1.z+=self.unit_sideLength
                                vertex1_Index.z+=1
                            if(edge_index==1 or edge_index==5 or edge_index==9 or edge_index==10):
                                pos1.x+=self.unit_sideLength
                                vertex1_Index.x+=1
                            pos2=pos1
                            vertex2_Index=vertex1_Index
                            # correct
                            if(edge_index>7 and edge_index<12):
                                pos2.y+=self.unit_sideLength
                                vertex2_Index.y+=1
                            elif(edge_index==1 or edge_index==3 or edge_index==5 or edge_index==7):
                                pos2.z+=self.unit_sideLength
                                vertex2_Index.z+=1
                            else: # 0,2,4,6
                                pos2.x+=self.unit_sideLength
                                vertex2_Index.x+=1
                        v1,v2=self.cube_isovalue[vertex1_Index],self.cube_isovalue[vertex2_Index]
                        factor=max(0,min(1,(1-v1)/(v2-v1)))
                        vertexPos=pos1+(pos2-pos1)*factor
                        self.vertex_pos[vertex_index+point_num]=vertexPos
                        if(ti.static(self.has_normal)):
                            n1,n2=self.cube_grad[vertex1_Index],self.cube_grad[vertex2_Index]
                            normal=n1+(n2-n1)*factor
                            self.vertex_normal[vertex_index+point_num]=normal.normalized(1e-5)


