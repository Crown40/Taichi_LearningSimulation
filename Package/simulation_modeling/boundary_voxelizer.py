import taichi as ti
import math


@ti.data_oriented
class Voxelizer:
    def __init__(self, dim=3, res=512, space_sideLength=1.0, radius=None, is_mesh=False):
        # length
        self.unit_sideLength=space_sideLength/res
        self.radius=radius if(radius!=None) else self.unit_sideLength
        occur_range=int(self.radius/self.unit_sideLength)
        self.occur_ndrange=[[-occur_range,occur_range+1] for _ in range(dim)]
        self.neighbor_ndrange=[[-1,2] for _ in range(dim)]
        # layout
        self.dim=dim
        self.res=res
        sqrt_res=int(math.sqrt(res))
        indices=[ti.ij,ti.ijk][dim-2]
        self.grid_size=occur_range*(2**dim) #2**dim
        self.grid_size=sqrt_res if(self.grid_size>sqrt_res) else self.grid_size 
        self.space_size=int(res//self.grid_size)
        # g_boundary
        self.boundary_pAmount=ti.field(int)
        self.boundary_gID=ti.Vector.field(dim,int)
        # space,area,page,block,grid,cell,unit,individual 
        self.space=ti.root.pointer(indices,self.space_size)
        self.grid=self.space.bitmasked(indices,self.grid_size) # dense|bitmasked|pointer
        # g_voxel_status
        self.voxel_status=ti.field(int)
        self.voxel_pAmount=ti.field(int)
        #self.grid.place(self.voxel_pAmount)
        self.grid.place(self.voxel_status,self.voxel_pAmount)
        # g_boundaries
        self.voxels_amount=ti.field(int,())
        self.boundary_amount=ti.field(int,())
        self.boundary_size=[1,2][dim-2]*(self.res)**(dim-1)
        print('v.boundaryVoxel_size:'+str(self.boundary_size))
        ti.root.dense(ti.i,self.boundary_size).place(self.boundary_gID,self.boundary_pAmount)
        # up_threshold
        self.fake_upThreshold=3**dim
        self.real_upThreshold=3**dim
        # Mesh
        if(is_mesh):
            self.vertices_datasize=[1,2][dim-2]*[4,24][dim-2]*res**2
            print('v.vertices_datasize:'+str(self.vertices_datasize))
            self.vertices_pos=ti.Vector.field(dim,float,self.vertices_datasize)
            self.vertices_amount=ti.field(int,())
            self.faces_amount=ti.field(int,())
            if(dim==2):
                self.quad_ndrange=[[0,2] for _ in range(dim)]
            else:
                self.xy_ndrange,self.yz_ndrange,self.zx_ndrange=[[0,2],[0,2],[0,1]],[[0,1],[0,2],[0,2]],[[0,2],[0,1],[0,2]]
                self.yx_ndrange,self.zy_ndrange,self.xz_ndrange=[[-1,1],[-1,1],[0,1]],[[0,1],[-1,1],[-1,1]],[[-1,1],[0,1],[-1,1]]

    # status:0(empty|outside),1(occur|inside),2(boundary-line),3(boundary-inside)
    def voxelize(self,pos):
        self.clear_all()
        self.determineBoundary(pos)
        self.space.deactivate_all()
        self.voxelizeBoundary(self.voxels_amount[None])
        
    def clear_all(self):
        self.space.deactivate_all()
        # self.voxel_boundaries.deactivate_all()

    @ti.kernel
    def determineBoundary(self,pos:ti.template()):
        # alias
        voxel_status,voxel_pAmount=ti.static(self.voxel_status,self.voxel_pAmount)
        # clear voxelsID
        self.voxels_amount[None]=0
        # scatter
        for P in ti.grouped(pos):
            I=int(pos[P]/self.unit_sideLength)
            voxel_pAmount[I]+=1
            for Offset in ti.grouped(ti.ndrange(*self.occur_ndrange)):
                J=I+Offset
                voxel_status[J]=1
        # filter 0-P-index
        zeroP_gIndex=int(pos[0]/self.unit_sideLength)
        voxel_pAmount[zeroP_gIndex]-=1  
                
        # determine fake-boundary
        for I in ti.grouped(voxel_status):
            if(voxel_status[I]==1):
                connect_count=0
                for Offset in ti.grouped(ti.ndrange(*self.neighbor_ndrange)):
                    J=I+Offset
                    connect_count+=(voxel_status[J]!=0)
                if(connect_count<self.fake_upThreshold):
                    voxel_status[I]=2
        
        # delete fake-boundary
        for I in ti.grouped(voxel_status):
            if(voxel_status[I]==2 and voxel_pAmount[I]==0):
                #voxel_status[I]=0
                for Offset in ti.grouped(ti.ndrange(*self.occur_ndrange)):
                    J=I+Offset
                    if(voxel_status[J]!=0 and voxel_pAmount[J]==0):
                        voxel_status[J]=0

        # determine real-boundary
        for I in ti.grouped(voxel_status):
            if(voxel_status[I]==1 and voxel_pAmount[I]!=0):
                connect_count=0
                for Offset in ti.grouped(ti.ndrange(*self.neighbor_ndrange)):
                    J=I+Offset
                    connect_count+=(voxel_status[J]!=0)
                if(connect_count<self.real_upThreshold):
                    voxel_status[I]=2
                    i=ti.atomic_add(self.voxels_amount[None],1) # self.dim
                    self.boundary_gID[i]=I
                    self.boundary_pAmount[i]=self.voxel_pAmount[I]
                    
        self.boundary_amount[None]=self.voxels_amount[None]
        # determine inside-boundary
        for I in ti.grouped(voxel_status):
            if(voxel_status[I]==2):
                for Offset in ti.grouped(ti.ndrange(*self.neighbor_ndrange)):
                    J=I+Offset
                    if(voxel_status[J]==1):
                        voxel_status[J]=3
        for I in ti.grouped(voxel_status):
            if(voxel_status[I]==3):
                i=ti.atomic_add(self.voxels_amount[None],1)
                self.boundary_gID[i]=I
                self.boundary_pAmount[i]=self.voxel_pAmount[I] #0
        pass
    
    @ti.kernel
    def voxelizeBoundary(self,length:int):
        I=ti.Vector.zero(int,self.dim)
        for i in range(length):
            I=self.boundary_gID[i]
            self.voxel_status[I]=(i<self.boundary_amount[None])+1
            self.voxel_pAmount[I]=self.boundary_pAmount[i]
        pass

    @ti.func
    def quading(self,Index,vertex_id,quad_ndrange):
        for vertex_Offset in ti.grouped(ti.ndrange(*quad_ndrange)):
                    point_pos=(Index+vertex_Offset)*self.unit_sideLength
                    self.vertices_pos[vertex_id]=point_pos
                    vertex_id+=1
        temp=self.vertices_pos[vertex_id-1]
        self.vertices_pos[vertex_id-1]=self.vertices_pos[vertex_id-2]
        self.vertices_pos[vertex_id-2]=temp
        return vertex_id

    @ti.kernel
    def meshing(self):
        self.vertices_amount[None]=0
        for I in ti.grouped(self.voxel_status): 
            if(self.voxel_status[I]==2):
                if(ti.static(self.dim==2)):
                    vertex_id=ti.atomic_add(self.vertices_amount[None],4)
                    self.quading(I,vertex_id,self.quad_ndrange)
                else: #dim==3
                    vertex_id=ti.atomic_add(self.vertices_amount[None],24)
                    J=I
                    vertex_id=self.quading(J,vertex_id,self.xy_ndrange)
                    vertex_id=self.quading(J,vertex_id,self.yz_ndrange)
                    vertex_id=self.quading(J,vertex_id,self.zx_ndrange)
                    #
                    J+=[1,1,1]
                    vertex_id=self.quading(J,vertex_id,self.yx_ndrange)
                    vertex_id=self.quading(J,vertex_id,self.zy_ndrange)
                    vertex_id=self.quading(J,vertex_id,self.xz_ndrange)
        #
        self.faces_amount[None]=int(self.vertices_amount[None]//4)

