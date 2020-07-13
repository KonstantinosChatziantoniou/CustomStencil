if !isdefined(Main, :CustomStencil)
    include("../module/CustomStencil.jl")
    using Main.CustomStencil
end
include("misc.jl")
coefs = Float64.([1 1 1])


thumbstack_stencil = @def_stencil_expression(
            @sum(i,-2,2,(
                @sum(j,-2,2, c[max(abs(i),abs(j))]*D[x+i,y+j,z])
            )) + @sum(i,1,2, c[i]*(D[x,y,z+i]+D[x,y,z-i]))
)

st_inst = NewStencilInstance(thumbstack_stencil, 2, coefs)


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

function cpu_thumbstack_stencil(data, radius,coefs)
    out = zeros(Float32,size(data))
    for i = (radius+1):(size(data,1)-radius)
        for j = (radius+1):(size(data,2)-radius)
            for k = (radius+1):(size(data,3)-radius)
                c = 0#coefs[1]*data[i,j,k]
                for rx = -radius:radius, ry = -radius:radius
                    cr = max(abs.([rx,ry])...)
                    c += coefs[cr+1]*(data[i+rx,j+ry,k])
                end
                for rz = 1:radius
                    c += coefs[rz+1]*(data[i,j,k+rz] + data[i,j,k-rz])
                end
                out[i,j,k] = c
            end
        end
    end
    return out
end


cout = cpu_thumbstack_stencil(data, radius, coefs)
for i = 2:t_steps
    global cout
    cout = cpu_thumbstack_stencil(cout, radius, coefs)
end



@assert(sum(abs.(cout .- o))./(reduce(*, size(data))) < 0.1)
mean_err = sum(abs.(cout .- o))./(reduce(*, size(data)))
# using Plots
# shift = 6
# heatmap(cout[:,:,Int(round(dz/2))+shift])
# heatmap(o[:,:,Int(round(dz/2))+shift])
#
# heatmap(o[:,:,Int(round(dz/2))+shift] .- cout[:,:,Int(round(dz/2))+shift])
#
#
# diff =  abs.(cout .- o)#./(reduce(*, size(data)))
# findmax(diff)
#
