using CUDA
include("./types")

mutable struct exec_config
    pad::Int
    single_step_offset::Int
    has_multi_step::Bool
    m_step::Int

end

function config(st_inst, data)
    radius = st_inst.radius
    if st_inst.combined_time_step != false
        ctp = st_inst.combined_time_step
        radius = radius*(2^(ctp-1))

end

function apply_single_step()



    conf = config()
    i = 0

    while  i < t_steps
        ## Assign the single time step kernel
        foo = st_inst.kernel
        offset = conf.single_step_offset
        time_inc = 1
        ## If multistep feasible, assign the m_step kernel
        if st_inst.combined_time_step != false
            if i + st_inst.combined_time_step <= t_steps
                foo = st_inst.m_kernel
                offset = 0
                time_inc = conf.m_step
            end
        end
        ## Construct arguments for kenerl
        args = (dev_data, dev_out, 0, 0, offset, offset)
        if st_inst.uses_vsq
           args = (args..., dev_vsq)
        end
        ## Kernel call
        @cuda(blocks=(bx,by,1), threads=(bdimx,bdimy),
                   shmem=((bdimx+2*radius)*(bdimy+2*radius))*sizeof(Float32),
                   foo(args...))
        dev_data,dev_out = dev_out,dev_data
        println("t = $(i), (bx,by) = $((bx,by))")
        i += 1
    end
end

function apply_multi_step()


end
