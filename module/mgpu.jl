using CUDA
using PaddedViews
using TimerOutputs

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

function comm_groups(n)
    gs = []

    for i = 1:n÷2
        @show id = i*2
        push!(gs, (id-1,id))
    end

    for i = 1:(n-1)÷2
        @show id = i*2 + 1
        push!(gs, (id-1,id))
    end
    return gs
end

function split_data_to_gpus(n_gpus,data, t_group, radius)
    ## Split data evenly
    len = round(size(data,3)/n_gpus)
    edges = []
    start = 1
    for i = 1:(n_gpus-1)
        push!(edges, [start,start+len-1])
        start += len
    end
    push!(edges, [start,size(data,3)])
    ed = deepcopy(edges)
    ## Determine the data with padding of the data
    pad = t_group*radius
    if n_gpus == 1
        return Int.(edges[1]), ed
    else
        (edges[1])[2] = (edges[1])[2] + pad
        for i = 2:(n_gpus-1)
            (edges[i])[1] = (edges[i])[1] - pad
            (edges[i])[2] = (edges[i])[2] + pad
        end
        (edges[n_gpus])[1] = (edges[n_gpus])[1] - pad
    end
    ## Determine the padding length for transfer
    [[-pad*(i!=1), pad*(i!=n_gpus)]  for i = 1:n_gpus]
    return map(x -> Int.(x) , edges),map(x -> Int.(x) , ed)
end

function ApplyMultiGPU(ngpus, st_inst, t_steps, data ;vsq=nothing, t_group=1, dbg=false)
    to = TimerOutput()
    #init_gpu_channels(ngpus)
    radius = max(st_inst.front_z_max, st_inst.behind_z_max)
    ## Split Data among gpus

    comm_g = comm_groups(ngpus)
    s_ind, u_ind = split_data_to_gpus(ngpus, data, t_group, radius)
    s_data = []
    padded_data = PadData(t_group*radius, data)
    for i = 1:ngpus
        s = s_ind[i]
        push!(s_data, @view padded_data[:,:,s[1]:s[2]])
    end
    @timeit to "Init" begin
    ## Create Unified for CPU
    buf_host = Mem.alloc(Mem.Unified, prod(size(padded_data))*sizeof(Float32))
    CUDA.Mem.advise(buf_host, Mem.ADVISE_SET_PREFERRED_LOCATION,
            prod(size(padded_data))*sizeof(Float32), device=CUDA.DEVICE_CPU)
    ptr_host = buf_host.ptr
    g_out = unsafe_wrap(Array{Float32, 3},
                        convert(Ptr{Float32}, buf_host),
                        size(padded_data), own=false)
    copyto!(g_out, padded_data)

    ## Create Unified GPU arrays
    gpu_buffers = [Mem.alloc(Mem.Unified, prod(size(s_data[i]))*sizeof(Float32))
                            for i = 1:ngpus]
    # for i = 1:ngpus
    #     CUDA.Mem.advise(gpu_buffers[i], Mem.ADVISE_SET_PREFERRED_LOCATION,
    #             prod(size(s_data[i]))*sizeof(Float32), device=CUDA.devuce_gpu))
    # end
    gpu_arrays_in =  [unsafe_wrap(CuArray{Float32, 3},
                        convert(CuPtr{Float32}, gpu_buffers[i]),
                        (size(padded_data,1),size(padded_data,2),s_ind[i][2]-s_ind[i][1]+1),
                        own=false) for i = 1:ngpus]
    gpu_pointers_in = [convert(CuPtr{Nothing}, gpu_arrays_in[i].ptr) for i = 1:ngpus]

    gpu_buffers_out = [Mem.alloc(Mem.Unified, prod(size(s_data[i]))*sizeof(Float32))
                            for i = 1:ngpus]
    gpu_arrays_out =  [unsafe_wrap(CuArray{Float32, 3},
                        convert(CuPtr{Float32}, gpu_buffers_out[i]),
                        (size(padded_data,1),size(padded_data,2),s_ind[i][2]-s_ind[i][1]+1),
                        own=false) for i = 1:ngpus]
    gpu_pointers_out = [convert(CuPtr{Nothing}, gpu_arrays_out[i].ptr) for i = 1:ngpus]

    for i = 1:ngpus
        CUDA.cuMemsetD32_v2(gpu_pointers_out[i], Float32(0), prod(size(gpu_arrays_out[i])))
    end
    ## MemCpy Initial Data
    @sync begin
        for i = 1:ngpus
            s = s_ind[i]
            h_ptr = ptr_host + (s[1]-1)*size(padded_data, 1)*size(padded_data,2)*sizeof(Float32)
            bcount = (s[2]-s[1]+1)*size(padded_data, 1)*size(padded_data,2)*sizeof(Float32)
            @async CUDA.cuMemcpy(gpu_pointers_in[i], h_ptr, bcount)
        end
    end
    end
    at_out = [true for i = 1:ngpus]
    t_counter = 0
    flag_break = false
    while true
        if t_counter + t_group >= t_steps
            t_group = t_steps - t_counter
            flag_break = true
        end
        ## kernel loop
        @timeit to "kernels" begin
        @sync begin
            for i = 1:ngpus
                @async begin
                    device!(i-1)
                    (dbg) && println(i, " start kernel")
                    j = i#(i+1)%ngpus + 1
                    at_out[i] = call_kernel(st_inst.bdim, i, ngpus,
                        gpu_arrays_in[j], gpu_arrays_out[j], t_group, st_inst)
                    (dbg) && println(i, " end kernel")
                end
            end
        end
        end
        if !at_out[1]
            gpu_arrays_in,gpu_arrays_out = gpu_arrays_out,gpu_arrays_in
            gpu_pointers_in,gpu_pointers_out = gpu_pointers_out,gpu_pointers_in
        end
        if flag_break
            break
        end
        # ## comm data
        bsize = size(gpu_arrays_in[1], 1)*size(gpu_arrays_in[1], 1)*
                    radius*t_group*sizeof(Float32)
        @timeit to "comms" begin
        @sync begin
            for i in comm_g
                @async begin
                    dx,dy,dz = size(gpu_arrays_in[i[1]])
                    p1 = gpu_pointers_in[i[1]]
                    p2 = gpu_pointers_in[i[2]]
                    offp1 = p1 + dx*dy*(dz-2radius*t_group)*sizeof(Float32)
                    CUDA.cuMemcpyDtoD_v2(p2, offp1, bsize)

                    offp1 = p1 + dx*dy*(dz-radius*t_group)*sizeof(Float32)
                    offp2 = p2 + dx*dy*(radius*t_group)*sizeof(Float32)
                    CUDA.cuMemcpyDtoD_v2(offp1, offp2, bsize)
                end
            end
        end
        end
        t_counter += t_group
    end
    ## Copy data to cpu
    @timeit to "download" begin
    @sync begin
        for i = 1:ngpus
            @async begin
                @show s = u_ind[i]
                @show host_off = size(g_out, 1)*size(g_out,2)*(s[1]-1)*sizeof(Float32)
                @show dev_off = (i!=1)*radius*t_group*size(g_out, 1)*size(g_out,2)*sizeof(Float32)
                @show bsize =  size(g_out, 1)*size(g_out,2)*(s[2]-s[1]+1)*sizeof(Float32)
                @info (s[1]-1, s[2]-s[1]+1)
                CUDA.cuMemcpy(ptr_host+host_off, gpu_pointers_in[i] + dev_off,
                            bsize)
            end
        end
    end
    end
    for i = 1:ngpus
        CUDA.Mem.free(gpu_buffers[i])
        CUDA.Mem.free(gpu_buffers_out[i])
    end
    println(to)
    return @view g_out[radius*t_group+1:end-radius*t_group,
            radius*t_group+1:end-radius*t_group, :]
end

function call_kernel(bdim, id, ngpus, dev_in, dev_out, t_steps, st_inst)
    bdimx = bdim
    bdimy = bdim
    radius = st_inst.max_radius
    dx = size(dev_in,1)
    dy = size(dev_in,2)
    bx = Int(floor((dx - 2*radius)/bdim))
    by = Int(floor((dy - 2*radius)/bdim))
    at_out = true
    for t = 0:t_steps-1
        offz_f = (id != 1)*t*radius
        offz_b = (id != ngpus)*t*radius
        args = (dev_in, dev_out, offz_f,offz_b, 0, 0)
        @cuda(blocks=(bx,by,1), threads=(bdimx,bdimy),
                    shmem=((bdimx+2*radius)*(bdimy+2*radius))*sizeof(Float32),
                    st_inst.kernel(args...))
        dev_in,dev_out = dev_out,dev_in
        at_out = !at_out
        @show "kernel $t $id $ngpus"
    end

    return at_out
end
