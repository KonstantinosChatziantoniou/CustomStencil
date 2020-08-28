if !isdefined(Main, :CustomStencil)
    include("../module/CustomStencil.jl")
    using Main.CustomStencil
end
include("misc.jl")
include("cpu_stencils.jl")


nx = 5
ny = 5
nz = 5
dx = 1<<(nx)
dy = 1<<(ny)
dz = 1<<(nz)

original_data = CreateData(dx,dy,dz)
## Stencil Definition

i = 1
radius = 1
t_steps = 2
coefs = [Float64(round(1/i, digits=2)) for i = 1:2]
star_stencil = @def_stencil_expression c[0]D[x,y,z] + @sum(i, 1, 1, c[i]*(
            D[x+i,y,z] + D[x,y+i,z] + D[x,y,z+i] +
            D[x-i,y,z] + D[x,y-i,z] + D[x,y,z-i]))
st_def = CreateStencilDefinition(star_stencil, coefs)


st_v = @def_stencil_expression v*c[0]D[x,y,z] + @sum(i, 1, 1, v*c[i]*(
            D[x+i,y,z] + D[x,y+i,z] + D[x,y,z+i] +
            D[x-i,y,z] + D[x,y-i,z] + D[x,y,z-i]))
st_v_def = CreateStencilDefinition(st_v, coefs)
st_inst = NewStencilInstance(st_def,st_v_def)

## Input Data size Definition
data = PadData(radius, original_data)
@time dout = ApplyStencil(st_inst, data, t_steps)
gpu_out = Array(dout)


using SymEngine

d = st_inst.stencil_sym
total_max_radius = st_inst.max_radius
radius = total_max_radius
max_radius = radius
Dsym = [symbols("w$(i)$(j)$(k)") for i = 1:(2total_max_radius+1),
                                    j = 1:(2total_max_radius+1),
                                    k = 1:(2total_max_radius+1)]

Doff = [[i,j,k] for i = -radius:radius, j=-radius:radius, k=-radius:radius]
new_Dsym = [symbols("w$(i)$(j)$(k)") for i = 1:(4total_max_radius+1),
                                    j = 1:(4total_max_radius+1),
                                    k = 1:(4total_max_radius+1)]
new_Doff =  [[i,j,k] for i = -2radius:2radius, j=-2radius:2radius, k=-2radius:2radius]

new_eq = 0
for i = 1:(2*max_radius+1), j = 1:(2*max_radius+1),k = 1:(2*max_radius+1)
    global d, Doff, Dsym, new_Dsym, new_Doff,new_eq,radius
    if coeff(d,Dsym[i,j,k]) != 0
        @show Dsym[i,j,k]
        @show c1 = coeff(d,Dsym[i,j,k])
        for ii = 1:(2*max_radius+1), jj = 1:(2*max_radius+1),kk = 1:(2*max_radius+1)
            if coeff(d,Dsym[ii,jj,kk]) != 0
                @show c = coeff(d,Dsym[ii,jj,kk])
                @show off2 = Doff[ii,jj,kk] + [i;j;k] .+ radius
                new_eq += c1*c*new_Dsym[off2...]
            end
        end
    end

end
new_eq


d1 = [0 0.5 0; 0.5 1 0.5; 0 0.5 0]
c = PaddedView(0, d1, (1:7,1:7), (3:5,3:5) )
c = Array(c)
res = zeros(Float64, 7 , 7)
for i = 1:3
    global res
    for j = 1:3
        res[i:i+2,j:j+2] += c[i+2,j+2]*d1

    end
end

@show res

SymEngine.

eq = a^2 + a + 2

for i in enumerate(eq)
    println(i)
end

SymEngine.subs(eq, b, symbols(c))
