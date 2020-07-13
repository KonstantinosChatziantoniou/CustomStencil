if !isdefined(Main, :CustomStencil)
    include("../module/CustomStencil.jl")
    using Main.CustomStencil
end
include("misc.jl")
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
#
# star_stencil = @def_stencil_expression c[0]D[x,y,z] + @sum(i, 1, 4, c[i]*(
#             D[x+i,y,z] + D[x,y+i,z] + D[x,y,z+i] +
#             D[x-i,y,z] + D[x,y-i,z] + D[x,y,z-i]))

st_inst = NewStencilInstance(coefs)


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
