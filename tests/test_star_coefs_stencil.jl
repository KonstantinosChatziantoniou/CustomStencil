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

st_inst = CustomStencil.NewStencilInstance(coefs)


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


function cpu_star_stencil(data, radius,coefs)
    out = zeros(Float32,size(data))
    for i = (radius+1):(size(data,1)-radius)
        for j = (radius+1):(size(data,2)-radius)
            for k = (radius+1):(size(data,3)-radius)
                c = coefs[1]*data[i,j,k]
                for r = 1:radius
                    c += coefs[r+1]*(data[i + r,j,k] + data[i - r,j,k] +
                    data[i,j + r,k] + data[i,j - r,k] +
                    data[i,j,k + r] + data[i,j,k - r])
                end
                println(c)
                out[i,j,k] = c
            end
        end
    end
    return out
end

c_coefs = Float64.([1 1 1 1 1])
cout = cpu_star_stencil(data, radius, c_coefs)
sum(cout)
for i = 2:t_steps
    global cout
    cout = cpu_star_stencil(cout, radius, c_coefs)
end



@assert(sum(abs.(cout .- o))./(reduce(*, size(data))) < 0.001)
mean_err = sum(abs.(cout .- o))./(reduce(*, size(data)))

using Plots
shift = 4
heatmap(cout[:,:,Int(round(dz/2))+shift])
heatmap(o[:,:,Int(round(dz/2))+shift])
sum(cout)
heatmap(o[:,:,Int(round(dz/2))+shift] .- cout[:,:,Int(round(dz/2))+shift])


diff =  abs.(cout .- o)#./(reduce(*, size(data)))
findmax(diff)
