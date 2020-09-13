include("../module/CustomStencil.jl")
include("../misc/misc.jl")
include("../misc/cpu_stencils.jl")
## Star Stencil definition with radius = 4
coefs = [0.75;0.5]
star_stencil = @def_stencil_expression c[0]D[x,y,z] + @sum(i, 1,1, c[i]*(
            D[x+i,y,z] + D[x,y+i,z] + D[x,y,z+i] +
            D[x-i,y,z] + D[x,y-i,z] + D[x,y,z-i]))
st_def = CreateStencilDefinition(star_stencil, coefs)
st_inst = NewStencilInstance(st_def, m_step=false)

radius = 4
nx = 6
ny = 6
nz = 6
dx = 1<<(nx)
dy = 1<<(ny)
dz = 1<<(nz)

data = CreateData(dx,dy,dz)

t_steps = 10

g1 = ApplyStencil(st_inst, data, t_steps)

g2 = ApplyMultiGPU(2, st_inst, t_steps, data, t_group=2)


using Plots
(sum(abs.(g2 .- g1)))
max(abs.(g2 .- g1)...)
max(g2...)


ts = 2
ind = size(g2,3)÷2 + ts
heatmap(g2[:,:,ind] .- g1[:,:,ind])
heatmap(g2[:,:,ind])
heatmap(g1[:,:,ind])


for i = 1:size(g1, 3)
    er = sum(abs.(g1[:,:,i] .- g2[:,:,i]))
    if er > 1
        println("err ", i, " ", er )
    end


end
