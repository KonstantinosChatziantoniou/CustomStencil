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
    coefs[posi,mid,mid] = round(Float64(1/i),digits=2)
    coefs[negi,mid,mid] = round(Float64(1/i),digits=2)
    coefs[mid, posi, mid] = round(Float64(1/i),digits=2)
    coefs[mid, negi, mid] = round(Float64(1/i),digits=2)
    coefs[mid, mid, posi] = round(Float64(1/i),digits=2)
    coefs[mid, mid, negi] = round(Float64(1/i),digits=2)

end
mid = radius + 1
coefs[mid,mid,mid] = 1
st_def = CreateStencilDefinition(coefs, uses_vsq=true)
st_inst = NewStencilInstance(st_def)
## Input Data size Definition
radius = 4
nx = 8
ny = 8
nz = 8
dx = 1<<(nx)
dy = 1<<(ny)
dz = 1<<(nz)


t_steps = 4
## GPU Implementation
original_data = CreateData(dx,dy,dz)
data = PadData(radius, original_data)
vsq = ones(Float32, dx, dy, dz)
vsq[100:200,:,:] .= 0.1
dout = ApplyStencil(st_inst, data, t_steps,vsq=vsq)
gpu_out = Array(dout)
