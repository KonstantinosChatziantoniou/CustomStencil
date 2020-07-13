

function CreateData(dimx, dimy, dimz, radius; init=0, mid=1)
    pad_dimx = dimx + 2*radius
    pad_dimy = dimy + 2*radius
    pad_dimz = dimz + 2*radius

    midx = (Int(round(dimx/2 - dimx/4)):Int(round(dimx/2 + dimx/4))) .+ radius
    midy = (Int(round(dimy/2 - dimy/4)):Int(round(dimy/2 + dimy/4))) .+ radius
    midz = (Int(round(dimz/2 - dimz/4)):Int(round(dimz/2 + dimz/4))) .+ radius

    data = fill(Float32(init), (pad_dimx, pad_dimy, pad_dimz))
    data[midx,midy, midz] .= mid

    return data
end


function cpu_stencil(data, radius)
    out = zeros(Float32,size(data))
    for i = (radius+1):(size(data,1)-radius)
        for j = (radius+1):(size(data,2)-radius)
            for k = (radius+1):(size(data,3)-radius)
                c = 3.0*data[i,j,k]
                for r = 1:radius
                    c += (data[i + r,j,k] + data[i - r,j,k] +
                    data[i,j + r,k] + data[i,j - r,k] +
                    data[i,j,k + r] + data[i,j,k - r])
                end
                out[i,j,k] = c
            end
        end
    end
    return out
end

function ConvertToRowMajor(data, dimx, dimy, dimz)
    return permutedims(data, [2,1,3])[:]
end
nothing
