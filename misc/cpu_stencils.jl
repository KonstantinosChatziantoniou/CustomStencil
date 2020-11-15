"""
    Pads all the dimensions, in constrast with @PadData
    that pads only x y dims.
"""
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

function cpu_star_stencil(org_data, radius, coefs, t_steps)
    data = PadDataCpu(radius, org_data)
    data = Array(data)
    function stencil!(data, out, radius, coefs)
        #out = zeros(Float32,size(data))
        for i = (radius+1):(size(data,1)-radius)
            for j = (radius+1):(size(data,2)-radius)
                for k = (radius+1):(size(data,3)-radius)
                    c = coefs[1]*data[i,j,k]
                    for r = 1:radius
                        c += coefs[r+1]*(data[i + r,j,k] + data[i - r,j,k] +
                        data[i,j + r,k] + data[i,j - r,k] +
                        data[i,j,k + r] + data[i,j,k - r])
                    end
                    out[i,j,k] = c
                end
            end
        end
    end

    #cout = stencil(data, radius, coefs)
    cout = zeros(Float64, size(data))
    for t = 1:t_steps
        stencil!(data, cout, radius, coefs)
        data,cout = cout,data
    end

    return @view data[(radius+1):(size(org_data,1)+radius),
            (radius+1):(size(org_data,2)+radius),
            (radius+1):(size(org_data,3)+radius)]
end



function cpu_dense_stencil(org_data, radius, coefs,t_steps)
    data = PadDataCpu(radius, org_data)
    data = Array(data)

    #out = zeros(Float32, dim)
    function stencil!(data, out, radius, coefs)
        dim = size(data)
        for x = (radius+1):(dim[1]-radius)
            for y = (radius+1):(dim[2]-radius)
                for z = (radius+1):(dim[3]-radius)
                    res = coefs[(radius+1),(radius+1),(radius+1)]*data[x,y,z]
                    for ix = -radius:radius
                        for iy = -radius:radius
                            for iz = -radius:radius
                                if ix == 0 &&  iy == 0 &&  iz == 0
                                    continue
                                end
                                ci = coefs[ix+radius+1, iy+radius+1, iz+radius+1]
                                res += ci*data[x+ix,y+iy,z+iz]
                            end
                        end
                    end
                    out[x,y,z] = res
                end
            end
        end
    end
    cout = zeros(Float64, size(data))
    for t = 1:t_steps
        stencil!(data, cout, radius, coefs)
        data,cout = cout,data
    end

    return @view data[(radius+1):(size(org_data,1)+radius),
            (radius+1):(size(org_data,2)+radius),
            (radius+1):(size(org_data,3)+radius)]
end


function cpu_star_stencil_vsc(org_data, vsq, radius, coefs, t_steps)
        data = PadDataCpu(radius, org_data)
        data = Array(data)
        v = PadDataCpu(radius, vsq)


        function stencil(data, out, vsq, radius, coefs)
            #out = zeros(Float32,size(data))
            for i = (radius+1):(size(data,1)-radius)
                for j = (radius+1):(size(data,2)-radius)
                    for k = (radius+1):(size(data,3)-radius)
                        c = coefs[1]*data[i,j,k]
                        for r = 1:radius
                            c += coefs[r+1]*(data[i + r,j,k] + data[i - r,j,k] +
                            data[i,j + r,k] + data[i,j - r,k] +
                            data[i,j,k + r] + data[i,j,k - r])
                        end
                        out[i,j,k] = c*vsq[i,j,k]
                    end
                end
            end
            #return out
        end

        cout = zeros(Float64, size(data))
        for t = 1:t_steps
            stencil(data, cout, v, radius, coefs)
            data,cout = cout,data
        end
        return @view data[(radius+1):(size(org_data,1)+radius),
                (radius+1):(size(org_data,2)+radius),
                (radius+1):(size(org_data,3)+radius)]
end


function cpu_dense_stencil_vsc(org_data,vsc, radius, coefs,t_steps)
    data = PadDataCpu(radius, org_data)
    data = Array(data)
    vsq = PadDataCpu(radius, vsc)
    #out = zeros(Float32, dim)
    function stencil!(data, out, vsc, radius, coefs)
        dim = size(data)
        for x = (radius+1):(dim[1]-radius)
            for y = (radius+1):(dim[2]-radius)
                for z = (radius+1):(dim[3]-radius)
                    res = coefs[(radius+1),(radius+1),(radius+1)]*data[x,y,z]
                    for ix = -radius:radius
                        for iy = -radius:radius
                            for iz = -radius:radius
                                if ix == 0 &&  iy == 0 &&  iz == 0
                                    continue
                                end
                                ci = coefs[ix+radius+1, iy+radius+1, iz+radius+1]
                                res += ci*data[x+ix,y+iy,z+iz]
                            end
                        end
                    end
                    out[x,y,z] = vsc[x,y,z]*res
                end
            end
        end
    end
    cout = zeros(Float64, size(data))
    for t = 1:t_steps
        stencil!(data, cout, vsq, radius, coefs)
        data,cout = cout,data
    end

    return @view data[(radius+1):(size(org_data,1)+radius),
            (radius+1):(size(org_data,2)+radius),
            (radius+1):(size(org_data,3)+radius)]
end


function cpu_star_stencil_v_nv(org_data, vsq, radius, coefs, t_steps)
        data = PadDataCpu(radius, org_data)
        data = Array(data)
        v = PadDataCpu(radius, vsq)


        function stencil(data, out, vsq, radius, coefs)
            #out = zeros(Float32,size(data))
            for i = (radius+1):(size(data,1)-radius)
                for j = (radius+1):(size(data,2)-radius)
                    for k = (radius+1):(size(data,3)-radius)
                        c = coefs[1]*data[i,j,k]
                        for r = 1:radius
                            c += coefs[r+1]*(data[i + r,j,k] + data[i - r,j,k] +
                            data[i,j + r,k] + data[i,j - r,k] +
                            data[i,j,k + r] + data[i,j,k - r])
                        end
                        out[i,j,k] = c*vsq[i,j,k] + c
                    end
                end
            end
            #return out
        end

        cout = similar(data)
        for t = 1:t_steps
            stencil(data, cout, v, radius, coefs)
            data,cout = cout,data
        end

        return @view data[:,:,(radius+1):(size(org_data,3)+radius)]
end


function cpu_star_stencil_vsq_prev_val(org_data,vsq,
            radius, coefs, t_steps, prev_time)
    data = PadDataCpu(radius, org_data)
    data = Array(data)
    out = zeros(Float32,size(data))
    v = PadDataCpu(radius, vsq)


    function stencil(data, out, vsq, radius, coefs, pc)
        for i = (radius+1):(size(data,1)-radius)
            for j = (radius+1):(size(data,2)-radius)
                for k = (radius+1):(size(data,3)-radius)
                    #out[i,j,k] != 0 && println(pc*out[i,j,k])
                    c = coefs[1]*data[i,j,k]
                    for r = 1:radius
                        c += coefs[r+1]*(data[i + r,j,k] + data[i - r,j,k] +
                        data[i,j + r,k] + data[i,j - r,k] +
                        data[i,j,k + r] + data[i,j,k - r])
                    end
                    out[i,j,k] =c*vsq[i,j,k] + c + pc*out[i,j,k]
                end
            end
        end
        return out
    end

    out = stencil(data, out, v, radius, coefs, prev_time)
    for t = 2:t_steps
        data,out = out,data
        out = stencil(data, out, v, radius, coefs, prev_time)
    end

    #if t_steps%2 == 1
    return @view out[:,:,(radius+1):(size(org_data,3)+radius)]
    #else
    #    return @view data[:,:,(radius+1):(size(org_data,3)+radius)]
    #end

end


function cpu_jacobi_stencil(org_data, radius, coefs, t_steps)
    data = PadData(radius, org_data)
    data = Array(data)
    function stencil!(data, out, radius, coefs)
        #out = zeros(Float32,size(data))
        for i = (radius+1):(size(data,1)-radius)
            for j = (radius+1):(size(data,2)-radius)
                k = 1
                    c = coefs[1]*data[i,j,k]
                    for r = 1:radius
                        c += coefs[r+1]*(data[i + r,j,k] + data[i - r,j,k] +
                        data[i,j + r,k] + data[i,j - r,k])
                    end
                    out[i,j,k] = c

            end
        end
    end

    #cout = stencil(data, radius, coefs)
    cout = zeros(Float64, size(data))
    for t = 1:t_steps
        stencil!(data, cout, radius, coefs)
        data,cout = cout,data
    end

    return @view data[(radius+1):(size(org_data,1)+radius),
            (radius+1):(size(org_data,2)+radius),
            1]
end
