if !isdefined(Main, :CustomStencil)
    include("../module/CustomStencil.jl")
    using Main.CustomStencil
end
include("../module/CustomStencil.jl")
include("misc.jl")
include("cpu_stencils.jl")

## Stencil Definition
coefs = Float64.([1;0.5;0.25])
star_stencil = @def_stencil_expression c[0]D[x,y,z] + @sum(i, 1, 2, c[i]*(
            D[x+i,y,z] + D[x,y+i,z] + D[x,y,z+i] +
            D[x-i,y,z] + D[x,y-i,z] + D[x,y,z-i]))
st_def = CreateStencilDefinition(star_stencil, coefs)
st_inst = NewStencilInstance(st_def, m_step=2)

## Input Data size Definition
radius = 2
nx = 5
ny = 5
nz = 5
dx = 1<<(nx)
dy = 1<<(ny)
dz = 1<<(nz)


t_steps = 4
## GPU Implementation
data = CreateData(dx,dy,dz)

dout = ApplyStencil(st_inst, data, t_steps)
gpu_out = Array(dout)
gpu_out = gpu_out[3:38, 3:38, :]
## CPU Implementation
cpu_out =  cpu_star_stencil(data, radius, coefs, t_steps)

## Check Result
mean_er = sum(abs.(cpu_out .- gpu_out))./(reduce(*, size(gpu_out)))
@assert(sum(abs.(cpu_out .- gpu_out))./(reduce(*, size(gpu_out))) < 0.001)


using Plots


shift = 0
Int(round(dz/2))+shift
heatmap(cpu_out[:,:,Int(round(dz/2))+shift])
heatmap(gpu_out[:,:,Int(round(dz/2))+shift])


d2 = ApplyMultiGPU(4, st_inst, t_steps, data)
gpu_out  = d2
heatmap(gpu_out[:,:,Int(round(dz/2))+shift])
heatmap(gpu_out[:,:,Int(round(dz/2))+shift]-cpu_out[:,:,Int(round(dz/2))+shift])
## Visual Check
using Plots
shift = 0
Int(round(dz/2))+shift
heatmap(cpu_out[:,:,Int(round(dz/2))+shift])
heatmap(gpu_out[:,:,Int(round(dz/2))+shift])

for i = 1:size(original_data,3)

    mean_er = sum(abs.(cpu_out[:,:,i] .- gpu_out[:,:,i]))./(reduce(*, size(gpu_out[:,:,1])))
    if mean_er > 0.01
        println(i, " ", mean_er)
    end
end




st_def = @def_stencil_expression c[0]*D[x,y,z] + c[1]*D[x+1,y,z]
c = Float64.([1; 0.5])
st_inst = CreateStencilDefinition(st_def,c)
st = NewStencilInstance(st_inst, m_step=3)
st_sym, r = CombineStencils(st_inst)
new_sym = st_sym
for i = 2:3
    global new_sym
    new_sym = CombineTimeSteps(st_sym, 1, new_sym, i)
end
new_sym
