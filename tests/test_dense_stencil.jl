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
nx = 5
ny = 5
nz = 5
dx = 1<<(nx)
dy = 1<<(ny)
dz = 1<<(nz)


t_steps = 5
## Actual time steps
data = CreateData(dx,dy,dz,radius)
dout = ApplyStencil(st_inst, data, t_steps)
#heatmap(dout1[:,:,Int(round(dz/2))])
o = Array(dout)

function cpu_dense_stencil(data, radius, coefs)
    dim = size(data)
    out = zeros(Float32, dim)
    for x = (radius+1):(dim[1]-radius)
        for y = (radius+1):(dim[2]-radius)
            for z = (radius+1):(dim[3]-radius)
                res = coefs[1]*data[x,y,z]
                for ix = -radius:radius
                    for iy = -radius:radius
                        for iz = -radius:radius
                            if ix == 0 &&  iy == 0 &&  iz == 0
                                continue
                            end
                            ci = max(abs.([ix, iy, iz])...)
                            ci = coefs[ci+1]
                            res += ci*data[x+ix,y+iy,z+iz]
                        end
                    end
                end
                out[x,y,z] = res
            end
        end
    end
    return out
end


cout = cpu_dense_stencil(data, radius, coefs)
for i = 2:t_steps
    global cout
    cout = cpu_dense_stencil(cout, radius, coefs)
end

@assert(sum(abs.(cout .- o))./(reduce(*, size(data))) < 0.001)
mean_err = sum(abs.(cout .- o))./(reduce(*, size(data)))

#
# using Plots
# shift = -13
# heatmap(cout[:,:,Int(round(dz/2))+shift])
# heatmap(o[:,:,Int(round(dz/2))+shift])
#
# shift = -14
# heatmap(o[:,:,Int(round(dz/2))+shift] .- cout[:,:,Int(round(dz/2))+shift])
#
# diff =  abs.(cout .- o)#./(reduce(*, size(data)))
# findmax(diff)
# sum(diff)./(reduce(*, size(data)))
