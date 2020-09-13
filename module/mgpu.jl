using CUDA
using TimerOutputs

function CudaHostMalloc(x,y,z)
    b = CUDA.Mem.alloc(Mem.Host, x*y*z*sizeof(Float32))
    return unsafe_wrap(Array{Float32, 3},
                convert(Ptr{Float32}, b),
                (x,y,z), own=false), b
end

function CudaAsyncDownload(src::CuArray, dest::Array, stream; zoffset=Int(0))
    # if sizeof(dest) != sizeof(src)
    #         error("Array sizes don't match. $(sizeof(dest)) vs $(sizeof(src))")
    # end
    sz = sizeof(dest)
    offset = zoffset*size(dest,1)*size(dest,2)*sizeof(dest[1])
    #ctx = CuCurrentContext()
    #buf = CUDA.Mem.DeviceBuffer(convert(CuPtr{Nothing}, src.ptr+offset), sz)#, ctx)
    buf = convert(CuPtr{Nothing}, src.ptr+offset)#, ctx)
    CUDA.cuMemcpyDtoHAsync_v2(dest, buf, sz, stream)
    #@show ( src.ptr,  src.ptr+offset)
    #CUDA.cuMemcpyDtoH(dest, buf, sz)

end
function CudaAsyncUpload(src::Array, dest::CuArray, stream; zoffset=0)
    # if sizeof(dest) != sizeof(src)
    #     error("Array sizes don't match.")
    # end
    sz = sizeof(src)
    offset =  zoffset*size(src,1)*size(src,2)*sizeof(src[1])
    #ctx = CuCurrentContext()
    #@show ( dest.ptr,  dest.ptr+offset)
    #buf = CUDA.Mem.DeviceBuffer(convert(CuPtr{Nothing}, dest.ptr+offset), sz)#, ctx)
    buf = convert(CuPtr{Nothing}, dest.ptr+offset)
    CUDA.cuMemcpyHtoDAsync_v2(buf, src, sz,stream)
    #CUDA.cuMemcpyHtoD(buf, src, sz)
end

"""
    gpu_channels::Vector{Channel{Array{Float32}}}

Used to communicate the updated halos among the Tasks
"""
gpu_channels = nothing
gpu_streams = nothing
g_out = nothing
all_stamps = []
timerouts = nothing
function init_gpu_channels(n_gpus::Integer; tp=Array{Float32,3})
    global timerouts
    timerouts = [TimerOutput() for i = 1:n_gpus]
    global gpu_channels
    gpu_channels = Vector{Channel{tp}}(undef, n_gpus)

    for i = 1:n_gpus
        gpu_channels[i] = Channel{tp}(1)
    end
    global gpu_streams
    gpu_streams = [CuDefaultStream() for i = 1:(n_gpus+1)]
    gpu_streams = gpu_streams[2:(n_gpus+1)]
end
function communicate_halos(id, hf, hb)
    joinfirst = true
    joinsecond = id != length(gpu_channels)
    # if odd number of gpus.. last one doesn't communicate
    if length(gpu_channels)%2 == 1
        if id == length(gpu_channels)
            joinfirst = false
            @show joinsecond = true
        end
    end

    recf = nothing
    recb = nothing
    global gpu_channels
    a = gpu_channels
    ## First exchange
    group = (id+1)÷2 #[1,2] -> g1  [3,4]->g2
    low = id%2 == 1
    if joinfirst
        #println(id, " join first ", group, low)
        if low
            put!(a[id+1], copy(hb))
            recb = take!(a[id])
        else
            recf = take!(a[id])
            put!(a[id-1], copy(hf))
        end
    end
    group = (id+2)÷2 #[2,3] -> g2
    low = id%2 == 0
    if id == 1
        joinsecond = false
    end
    ## Second exchange
    if joinsecond
        #println(id, " join second ", group, low)
        if low
            put!(a[id+1], copy(hb))
            recb = take!(a[id])
        else
            recf = take!(a[id])
            put!(a[id-1], copy(hf))
        end
    end

    ## Copy to pageable mem
    if hf != nothing
        copy!(hf,recf)
    end
    if hb != nothing
        copy!(hb, recb)
    end

end
"""
    communicate_halos(id, halo_front, halo_behind)

Send and receive the halos to id +- 1.
`halo_front` or `halo_behind` can be `nothing`
"""
# function communicate_halos(id, halo_front, halo_behind)
#     global gpu_channels
#     a = gpu_channels
#     n_tasks = length(gpu_channels)
#     rec_halo_f = nothing
#     rec_halo_b = nothing
#     if id == 1
#        #@show "1 sends to 2"
#        #@info "1 sending to 2 $(sum(halo_behind))"
#        put!(a[2], copy(halo_behind))
#        #@show "1 recieves from 2"
#        rec_halo_b = take!(a[1])
#        #@info "1 received from 2 $(sum(rec_halo_b))"
#    elseif id == n_tasks
#
#        #@show "$(id) receives from $(id-1)"
#        rec_halo_f = take!(a[n_tasks])
#        #@info "$(id) received from $(id-1) $(sum(rec_halo_f))"
#        #@show "$(id) sends to $(id-1)"
#        #@info "$(id) sending to $(id-1) $(sum(halo_front))"
#        put!(a[n_tasks-1], copy(halo_front))
#    else
#        #@show "$(id) receives from $(id-1)"
#        rec_halo_f = take!(a[id])
#        #@show "$(id) sends to $(id-1)"
#        put!(a[id-1], copy(halo_front))
#        #@show "$(id) sends to $(id+1)"
#        put!(a[id+1], copy(halo_behind))
#        #@show "$(id) receives from $(id+1)"
#        rec_halo_b = take!(a[id])
#    end
#
#    #return rec_halo_f, rec_halo_b
#    if halo_front != nothing
#        copy!(halo_front,rec_halo_f)
#    end
#    if halo_behind != nothing
#        copy!(halo_behind, rec_halo_b)
#    end
# end
#     if id == 1
#         put!(a[2], halo_behind)
#         rec_halo_b = take!(a[1])
#     elseif id == n_tasks
#         rec_halo_f = take!(a[n_tasks])
#         put!(a[n_tasks-1], halo_front)
#     else
#         rec_halo_f = take!(a[id])
#         put!(a[id-1], halo_front)
#
#         put!(a[id+1], halo_behind)
#         rec_halo_b = take!(a[id])
#     end
#
#     return rec_halo_f, rec_halo_b
# end


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

function ApplyMultiGPU(n_gpus, st_inst, t_steps, data ;vsq=nothing, t_group=1)
    global g_out = zeros(Float32, size(data))
    init_gpu_channels(n_gpus)
    radius = max(st_inst.front_z_max, st_inst.behind_z_max)
    ## Split Data among gpus
    s_ind, u_ind = split_data_to_gpus(n_gpus,data, t_group, radius)
    s_data = []
    for i = 1:n_gpus
        s = s_ind[i]
        push!(s_data, @view data[:,:,s[1]:s[2]])
    end

    s_vsq = nothing
    if vsq != nothing
        s_vsq = []
        for i = 1:n_gpus
            s = s_ind[i]
            push!(s_vsq, @view vsq[:,:,s[1]:s[2]])
        end
    end
    # tasks = [Task(closure_constr(
    #             i, t_steps, t_group, u_ind[i], st_inst, s_data[i],s_vsq
    #             )) for i = 1:n_gpus]
    timerout = TimerOutput()
    # @timeit timerout "Global" begin
    #     NVTX.@range "main loop" begin
    #         t = schedule.(tasks)
    #         wait.(t)
    #     end
    # end
    tasks = [(closure_constr(
                i, t_steps, t_group, u_ind[i], st_inst, s_data[i],s_vsq
                )) for i = 1:n_gpus]
    @timeit timerout "Global" begin
        NVTX.@range "main loop" begin
            t = [Threads.@spawn i() for i in tasks]
            wait.(t)
        end
    end
    println(timerout)
    global timerouts
    for i in timerouts
        println(i)
    end
    global g_out
    return g_out
end

function closure_constr(id, t_steps, t_group, save_ind, st_inst, org_data, vsq)
    #id, t_steps, t_group, save_ind, st_inst, data, vsq
    return function one_gpu()
        global gpu_channels
        global timerouts
        timerout = timerouts[id]
        #global gpu_streams
        #cstr = gpu_streams[id]
        d = id-1
        if length(collect(devices())) == 1
            @warn "Only on gpu"
            d = 0
        end
        device!(d)

        @timeit timerout "Init $id - $(Threads.threadid())" begin
            cstr = CUDA.CuStream()
            println(id, " using ", device())
            pers_t_group = t_group
            radius = st_inst.max_radius
            bdimx = st_inst.bdim
            bdimy = st_inst.bdim
            data = Array(PadData(radius, org_data))
            dx = size(data,1)
            dy = size(data,2)
            dz = size(data,3)
            #@show size(data)
            b_dev_a = Mem.alloc(Mem.Unified, prod(size(data))*sizeof(Float32))
            dev_data = unsafe_wrap(CuArray{Float32,3}, convert(CuPtr{Float32}, b_dev_a),
                      size(data); own=false)
            copyto!(dev_data, (data))
            #dev_out = CUDA.zeros(Float32, size(data))
            b_dev_b = Mem.alloc(Mem.Unified, prod(size(data))*sizeof(Float32))
            dev_out = unsafe_wrap(CuArray{Float32,3}, convert(CuPtr{Float32}, b_dev_b),
                      size(data); own=false)
            CUDA.cuMemsetD32_v2(dev_out,Float32(0),prod(size(data)))
            dev_vsq = nothing
            #yield()
            if st_inst.uses_vsq
                if vsq isa Nothing
                    error("vsq array not provided")
                end
                dev_vsq = CuArray(vsq)
            end
            bx = Int(floor((dx - 2*radius)/bdimx))
            by = Int(floor((dy - 2*radius)/bdimy))
            ## Do first loop
            #####################################################
            ## Allocate pinned memory for async mem transfer   ##
            #####################################################
            halo_f = nothing
            b_f = nothing
            halo_b = nothing
            b_b = nothing
            (id != 1) && ((halo_f,b_f) = CudaHostMalloc(dx,dy,radius*t_group))
            (id != length(gpu_channels)) && ((halo_b, b_b) = CudaHostMalloc(dx,dy,radius*t_group))
            #####################################################
            at_out = true
            t_counter = 0
            flag_break = false
        end
        @timeit timerout "whole loop" begin
        while true
            if t_counter + t_group >= t_steps
                t_group = t_steps - t_counter
                flag_break = true
            end
            ## Do t_group loop
            @timeit timerout "kernel loop $id" begin
                for t = 0:(t_group-1)
                    offz_f = (id != 1)*t*radius
                    offz_b = (id != length(gpu_channels))*t*radius
                    args = (dev_data, dev_out, offz_f,offz_b, 0, 0)
                    if st_inst.uses_vsq
                        args = (args..., dev_vsq)
                    end
                    @cuda(blocks=(bx,by,1), threads=(bdimx,bdimy),
                                shmem=((bdimx+2*radius)*(bdimy+2*radius))*sizeof(Float32),
                                stream=cstr,
                                st_inst.kernel(args...))
                                yield()
                    #println("t = $id $(t) $(t_counter + t)")
                    dev_data,dev_out = dev_out,dev_data
                    at_out = !at_out
                    #yield()
                end
                #CUDA.synchronize()
            end
            # if !at_out
            dev_data,dev_out = dev_out,dev_data
            #     at_out = !at_out
            # end
            if flag_break
                break
            end
            @timeit timerout "comms $id" begin
                ## Communicate and replace halos
                if halo_f != nothing
                   CudaAsyncDownload(dev_out,halo_f ,cstr, zoffset=radius*t_group)
                   #println(id, " halo_f ", sum(halo_f), " -- ", sum(dev_out[:,:,(radius*t_group+1):2radius*t_group]))
               end
               if halo_b != nothing
                   zofst = dz - 2*radius*t_group
                   CudaAsyncDownload(dev_out,halo_b, cstr, zoffset=zofst)
                   #println(id, " halo_b ", sum(halo_b), " -- ", sum(dev_out[:,:,(zofst+1):end]))
               end
               yield()
               communicate_halos(id, (halo_f), (halo_b))
           # #NVTX.@range "COM UP $id" begin
               if halo_f != nothing
                   #part_dev_arr = unsafe_wrap(CuArray{Float32, 3}, dev_out.ptr, (dx,dy,radius*t_group))
                   CudaAsyncUpload(halo_f, dev_out, cstr)
               end
               if halo_b != nothing
                   # part_dev_arr = unsafe_wrap(CuArray{Float32, 3},
                   #             dev_out.ptr + sizeof(Float32)*dx*dy*(dz-radius*t_group),
                   #             (dx,dy,radius*t_group))

                   zofst = dz - radius*t_group
                   CudaAsyncUpload(halo_b,dev_out, cstr, zoffset=zofst)
               end
               yield()
               #CUDA.synchronize()
           end
       # #end
           t_counter += t_group
           dev_data,dev_out = dev_out,dev_data
        end
    end
        ## SAVE dev_out
        println(id, " end using ", device())
        global g_out
        t_group = pers_t_group
        id, length(gpu_channels)
        g_out[:,:,save_ind[1]:save_ind[2]] = Array(dev_out)[
                            (radius+1):(size(org_data,1)+radius),
                            (radius+1):(size(org_data,2)+radius),
                            1+(id!=1)*(radius*t_group):end-(id!=length(gpu_channels))*(radius*t_group)]


        CUDA.Mem.free(b_dev_a)
        CUDA.Mem.free(b_dev_b)
        (halo_f != nothing) && CUDA.Mem.free(b_f)
        (halo_b != nothing) && CUDA.Mem.free(b_b)
    end
end
