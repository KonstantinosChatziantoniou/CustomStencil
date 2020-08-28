if !isdefined(Main, :CustomStencil)
    include("../module/CustomStencil.jl")
    using Main.CustomStencil
end
include("misc.jl")
include("cpu_stencils.jl")

## Stencil Definition
coefs = Float64.([1; 0.5; 0.25; 0.125; 0.06])

nv_star_stencil = @def_stencil_expression -c[0]D[x,y,z] + @sum(i, 1, 4, c[i]*(
            D[x+i,y,z] + D[x,y+i,z] + D[x,y,z+i] +
            D[x-i,y,z] + D[x,y-i,z] + D[x,y,z-i]))

star_stencil = @def_stencil_expression -v*c[0]D[x,y,z] + v*@sum(i, 1, 4, c[i]*(
            D[x+i,y,z] + D[x,y+i,z] + D[x,y,z+i] +
            D[x-i,y,z] + D[x,y-i,z] + D[x,y,z-i]))

v_def = CreateStencilDefinition(star_stencil, coefs)
nv_def = CreateStencilDefinition(nv_star_stencil, coefs)
st_inst = NewStencilInstance(v_def, nv_def)

## Input Data size Definition
radius = 4
nx = 5
ny = 5
nz = 5
dx = 1<<(nx)
dy = 1<<(ny)
dz = 1<<(nz)


t_steps = 4
## GPU Implementation
original_data = CreateData(dx,dy,dz)
data = PadData(radius, original_data)
vsq = ones(Float32, dx, dy, dz)
vsq[10:15,:,:] .= 0.1
dout = ApplyStencil(st_inst, data, t_steps,vsq=vsq)
gpu_out = Array(dout)


## CPU Implementation
cpu_out =  cpu_star_stencil_v_nv(original_data, vsq, radius, coefs, t_steps)

## Check Result
mean_er = sum(abs.(cpu_out .- gpu_out))./(reduce(*, size(gpu_out)))
@assert(sum(abs.(cpu_out .- gpu_out))./(reduce(*, size(gpu_out))) < 0.001)
## Visual Check
# using Plots
# shift = -10
# Int(round(dz/2))+shift
# heatmap(cout[:,:,Int(round(dz/2))+shift])
# heatmap(o[:,:,Int(round(dz/2))+shift])
