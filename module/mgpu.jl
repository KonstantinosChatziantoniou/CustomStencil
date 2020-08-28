using CUDA
"""
    gpu_channels::Vector{Channel{Array{Float32}}}

Used to communicate the updated halos among the Tasks
"""
gpu_channels = nothing
gpu_streams = nothing
g_out = nothing

function init_gpu_channels(n_gpus::Integer; tp=Array{Float32,3})
    global gpu_channels
    gpu_channels = Vector{Channel{tp}}(undef, n_gpus)

    for i = 1:n_gpus
        gpu_channels[i] = Channel{tp}(1)
    end
    global gpu_streams
    gpu_streams = [CuStream() for i = 1:(n_gpus+1)]
    gpu_streams = gpu_streams[2:(n_gpus+1)]
end

"""
    communicate_halos(id, halo_front, halo_behind)

Send and receive the halos to id +- 1.
`halo_front` or `halo_behind` can be `nothing`
"""
function communicate_halos(id, halo_front, halo_behind)
    global gpu_channels
    a = gpu_channels
    n_tasks = length(gpu_channels)
    rec_halo_f = nothing
    rec_halo_b = nothing
    if id == 1
        put!(a[2], halo_behind)
        rec_halo_b = take!(a[1])
    elseif id == n_tasks
        rec_halo_f = take!(a[n_tasks])
        put!(a[n_tasks-1], halo_front)
    else
        rec_halo_f = take!(a[id])
        put!(a[id-1], halo_front)

        put!(a[id+1], halo_behind)
        rec_halo_b = take!(a[id])
    end

    return rec_halo_f, rec_halo_b
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
    tasks = [Task(closure_constr(
                i, t_steps, t_group, u_ind[i], st_inst, s_data[i],s_vsq
                )) for i = 1:n_gpus]

    t = schedule.(tasks)
    wait.(t)
    global g_out
    return g_out
end

function closure_constr(id, t_steps, t_group, save_ind, st_inst, data, vsq)
    #id, t_steps, t_group, save_ind, st_inst, data, vsq
    return function one_gpu()
        global gpu_channels
        global gpu_streams
        pers_t_group = t_group
        radius = st_inst.max_radius
        bdimx = 32
        bdimy = 32
        dx = size(data,1)
        dy = size(data,2)
        dev_data = CuArray(data)
        dev_out = CUDA.zeros(Float32, size(data))
        dev_vsq = nothing
        if st_inst.uses_vsq
            if vsq isa Nothing
                error("vsq array not provided")
            end
            dev_vsq = CuArray(vsq)
        end
        bx = Int(floor((dx - 2*radius)/bdimx))
        by = Int(floor((dy - 2*radius)/bdimy))
        ## Do first loop

        at_out = true
        t_counter = 0
        flag_break = false
        while true
            if t_counter + t_group >= t_steps
                t_group = t_steps - t_counter
                flag_break = true
            end
            ## Do t_group loop
            for t = 0:(t_group-1)
                offz_f = (id != 1)*t*radius
                offz_b = (id != length(gpu_channels))*t*radius
                args = (dev_data, dev_out, offz_f,offz_b)
                if st_inst.uses_vsq
                    args = (args..., dev_vsq)
                end
                @cuda(blocks=(bx,by,1), threads=(bdimx,bdimy),
                            shmem=((bdimx+2*radius)*(bdimy+2*radius))*sizeof(Float32),
                            stream=gpu_streams[id],
                            st_inst.kernel(args...))
                println("t = $id $(t) $(t_counter + t)")
                dev_data,dev_out = dev_out,dev_data
                at_out = !at_out
            end
            # if !at_out
            dev_data,dev_out = dev_out,dev_data
            #     at_out = !at_out
            # end
            if flag_break
                break
            end
            ## Communicate and replace halos
            halo_f = nothing
            halo_b = nothing
            h_tf = (1 + radius*t_group):(1+2*radius*t_group)
            (id != 1) && (halo_f = Array(dev_out[:,:,h_tf]))
            h_tb = [(2*radius*t_group-1), radius*t_group]
            (id != length(gpu_channels)) && (halo_b = Array(dev_out[:,:,(end-h_tb[1]):end-h_tb[2]]))
            halo_f,halo_b = communicate_halos(id, (halo_f), (halo_b))

            (halo_f != nothing) && (dev_out[:,:,1:(radius*t_group)] = halo_f)
            (halo_b != nothing) && (dev_out[:,:,(end-radius*t_group+1):end] = halo_b)
            t_counter += t_group
            dev_data,dev_out = dev_out,dev_data
        end

        ## SAVE dev_out
        global g_out
        t_group = pers_t_group
        id, length(gpu_channels)
        g_out[:,:,save_ind[1]:save_ind[2]] = Array(dev_out[:,:,1+(id!=1)*(radius*t_group):end-(id!=length(gpu_channels))*(radius*t_group)])
    end
end
