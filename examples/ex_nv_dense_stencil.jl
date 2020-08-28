if !isdefined(Main, :CustomStencil)
    include("../module/CustomStencil.jl")
    using Main.CustomStencil
end
include("misc.jl")
include("cpu_stencils.jl")

## Stencil Definition
coefs = Float64.([0.25; 0.125; 0.06])
dense_stencil = @def_stencil_expression(
            @sum(i,-2,2,
                @sum(j,-2,2,
                    @sum(k,-2,2, c[max(abs(i),abs(j),abs(k))]*D[x+i,y+j,z+k]))))
st_def = CreateStencilDefinition(dense_stencil, coefs)
st_inst = NewStencilInstance(st_def)

## Input Data size Definition
radius = 2
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
dout = ApplyStencil(st_inst, data, t_steps)
gpu_out = Array(dout)
