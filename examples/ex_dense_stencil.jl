if !isdefined(Main, :CustomStencil)
    include("../module/CustomStencil.jl")
    using Main.CustomStencil
end
include("misc.jl")
coefs = [-5 0.1875 0.0625]


dense_stencil = @def_stencil_expression(
            @sum(i,-2,2,
                @sum(j,-2,2,
                    @sum(k,-2,2, c[max(abs(i),abs(j),abs(k))]*D[x+i,y+j,z+k]))))

st_inst = NewStencilInstance(dense_stencil, 2, coefs)


radius = 2
nx = 8
ny = 8
nz = 8
dx = 1<<(nx)
dy = 1<<(ny)
dz = 1<<(nz)


t_steps = 5
## Actual time steps
data = CreateData(dx,dy,dz,radius)
dout = ApplyStencil(st_inst, data, t_steps)
#heatmap(dout1[:,:,Int(round(dz/2))])
o = Array(dout)


isdefined

isdefined(Main, :CustomStencil)
