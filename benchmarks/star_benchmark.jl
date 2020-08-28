if !isdefined(Main, :CustomStencil)
    include("../module/CustomStencil.jl")
    using Main.CustomStencil
end
include("misc.jl")
include("cpu_stencils.jl")


nx = 9
ny = 9
nz = 8
dx = 1<<(nx)
dy = 1<<(ny)
dz = 1<<(nz)

original_data = CreateData(dx,dy,dz)
## Stencil Definition
t_cases = [32 16 8 4]
i = 16
radius = i
t_steps = Int(max(t_cases...)/i)
coefs = [Float64(round(1/i, digits=2)) for i = (radius+1):-1:1]
star_stencil = @def_stencil_expression c[0]D[x,y,z] + @sum(i, 1, 16, c[i]*(
            D[x+i,y,z] + D[x,y+i,z] + D[x,y,z+i] +
            D[x-i,y,z] + D[x,y-i,z] + D[x,y,z-i]))
st_def = CreateStencilDefinition(star_stencil, coefs)
st_inst = NewStencilInstance(st_def)

## Input Data size Definition
data = PadData(radius, original_data)
@time dout = ApplyStencil(st_inst, data, t_steps)
gpu_out = Array(dout)

# t_cases = [32 16 8 4]
# i = t_cases[2]
# radius = i
# t_steps = Int(max(t_cases...)/i)
# coefs = [Float64(round(1/i, digits=2)) for i = (radius+1):-1:1]
# star_stencil = @def_stencil_expression c[0]D[x,y,z] + @sum(i, 1, 16, c[i]*(
#             D[x+i,y,z] + D[x,y+i,z] + D[x,y,z+i] +
#             D[x-i,y,z] + D[x,y-i,z] + D[x,y,z-i]))
# st_def = CreateStencilDefinition(star_stencil, coefs)
# st_inst = NewStencilInstance(st_def)
#
# ## Input Data size Definition
# data = PadData(radius, original_data)
# @time dout = ApplyStencil(st_inst, data, t_steps)
# gpu_out = Array(dout)
#
# t_cases = [32 16 8 4]
i = t_cases[3]
radius = i
t_steps = Int(max(t_cases...)/i)
coefs = [Float64(round(1/i, digits=2)) for i = (radius+1):-1:1]
star_stencil = @def_stencil_expression c[0]D[x,y,z] + @sum(i, 1, 8, c[i]*(
            D[x+i,y,z] + D[x,y+i,z] + D[x,y,z+i] +
            D[x-i,y,z] + D[x,y-i,z] + D[x,y,z-i]))
st_def = CreateStencilDefinition(star_stencil, coefs)
st_inst = NewStencilInstance(st_def)

## Input Data size Definition
data = PadData(radius, original_data)
@time dout = ApplyStencil(st_inst, data, t_steps)
gpu_out = Array(dout)

t_cases = [32 16 8 4]
i = t_cases[4]
radius = i
t_steps = Int(max(t_cases...)/i)
coefs = [Float64(round(1/i, digits=2)) for i = (radius+1):-1:1]
star_stencil = @def_stencil_expression c[0]D[x,y,z] + @sum(i, 1, 4, c[i]*(
            D[x+i,y,z] + D[x,y+i,z] + D[x,y,z+i] +
            D[x-i,y,z] + D[x,y-i,z] + D[x,y,z-i]))
st_def = CreateStencilDefinition(star_stencil, coefs)
st_inst = NewStencilInstance(st_def)

## Input Data size Definition
data = PadData(radius, original_data)
@time dout = ApplyStencil(st_inst, data, t_steps)
gpu_out = Array(dout)
