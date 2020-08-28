if !isdefined(Main, :CustomStencil)
    include("../module/CustomStencil.jl")
    using Main.CustomStencil
end
include("misc.jl")
include("cpu_stencils.jl")

## Stencil Definition
radius = 4
l = 2*radius + 1
coefs = zeros(Float64, (l,l,l))

for i = 1:radius
    global radius
    posi = radius + i + 1
    negi = radius -i + 1
    mid = radius + 1
    coefs[posi,mid,mid] = 1
    coefs[negi,mid,mid] = 1
    coefs[mid, posi, mid] = 1
    coefs[mid, negi, mid] = 1
    coefs[mid, mid, posi] = 1
    coefs[mid, mid, negi] = 1

end
mid = radius + 1
coefs[mid,mid,mid] = 1
st_def = CreateStencilDefinition(coefs, uses_vsq=true)
st_inst = NewStencilInstance(st_def)
## Input Data size Definition
radius = 4
nx = 5
ny = 5
nz = 5
dx = 1<<(nx)
dy = 1<<(ny)
dz = 1<<(nz)


t_steps = 1
## GPU Implementation
original_data = CreateData(dx,dy,dz)
data = PadData(radius, original_data)
vsq = ones(Float32, dx, dy, dz)
vsq[10:15,:,:] .= 0.1
dout = ApplyStencil(st_inst, data, t_steps,vsq=vsq)
gpu_out = Array(dout)

## CPU Implementation
coefs = Float64.([1 1 1 1 1])
cpu_out =  cpu_star_stencil_v(original_data, vsq, radius, coefs, t_steps)

## Check Result
mean_er = sum(abs.(cpu_out .- gpu_out))./(reduce(*, size(gpu_out)))
@assert(sum(abs.(cpu_out .- gpu_out))./(reduce(*, size(gpu_out))) < 0.001)
## Visual Check
# using Plots
# shift = -10
# Int(round(dz/2))+shift
# heatmap(cout[:,:,Int(round(dz/2))+shift])
# heatmap(o[:,:,Int(round(dz/2))+shift])
