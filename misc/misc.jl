using PaddedViews

function CreateData(dimx, dimy, dimz; init=0, mid=1, rng=4)


    midx = (Int(round(dimx/2 - dimx/rng)):Int(round(dimx/2 + dimx/rng)))
    midy = (Int(round(dimy/2 - dimy/rng)):Int(round(dimy/2 + dimy/rng)))
    midz = 1
    if dimz > 1
        midz = (Int(round(dimz/2 - dimz/rng)):Int(round(dimz/2 + dimz/rng)))
    end
    data = fill(Float32(init), (dimx, dimy, dimz))
    data[midx,midy, midz] .= mid

    return data
end

function PadData(pd, data)
    x = size(data,1)
    y = size(data,2)
    z = size(data,3)
    padx = 1:(x+2pd)
    pady = 1:(y+2pd)
    padz = 1:z
    datax = (pd+1):(x+pd)
    datay = (pd+1):(y+pd)
    dataz = (1:z)
    return PaddedView(0, data, (padx,pady,padz), (datax, datay, dataz))

end

function PadDataCpu(pd,data)
    x = size(data,1)
    y = size(data,2)
    z = size(data,3)
    padx = 1:(x+2pd)
    pady = 1:(y+2pd)
    padz = 1:(z+2pd)
    datax = (pd+1):(x+pd)
    datay = (pd+1):(y+pd)
    dataz = (pd+1):(z+pd)
    return PaddedView(0, data, (padx,pady,padz), (datax, datay, dataz))
end

nothing
