#module CustomStencil
include("types.jl")
include("parsing.jl")
include("stencil_creation.jl")
include("kernel_gen.jl")
include("mgpu.jl")
include("fft_2d.jl")
include("fft_3d.jl")

using TimerOutputs
using StaticArrays
timers = Vector{TimerOutput}(undef,0)
using DSP

export @def_stencil_expression
export CreateStencilDefinition
export NewStencilInstance
export StencilInstance
export StencilDefinition
export ApplyStencil
export ApplyMultiGPU
@info collect(devices())
@info CUDA.version()
"""
    macro def_stencil_expression(expr::Expr)

Returns an expression with @sum macro and :abs :min: max functions
evaluated and replaced.
"""
macro def_stencil_expression(expr::Expr)#, data, current, t)
    expr = Meta.quot(expr)
    expr = eval_macro(expr, Symbol("@sum"))
    for i in [:abs, :max, :min]
        expr = eval_func!(expr, i)
    end
    return expr
end

function def_stencil_expression(expr::Expr)
    expr = eval_macro(expr, Symbol("@sum"))
    for i in [:abs, :max, :min]
        expr = eval_func!(expr, i)
    end
    return expr
end
```
    Used by eval to parse an expression to symbolic math
```
D = Array{Basic,1}(undef,1)
c = Array{Float64,1}(undef,1)
v = symbols("vsq_sym")


"""
    CreateStencilDefinition(expr::Expr, coefs::Union{Array{Float64,1}, Nothing})

Creates a Stencil definition struct given an expression and optinally a
coefficient matrix.
"""
function CreateStencilDefinition(expr::Expr, coefs::Union{Vector, Nothing})
    #println(expr)
    if coefs != nothing
        coefs = Float64.(coefs)
    end
    @show mn,mx = minmax_const(expr)
    m = mx > abs(mn) ? mx : abs(mn)
    if !(coefs isa Nothing)
        if (length(coefs)) == m + 1
            nothing
        elseif (length(coefs)) == 1000 + 2m + 1
            nothing
        else
            error("Coefficient matrix should have length of max radius + 1. is " ,length(coefs), " should be ", m +1)
        end
    end
    return StencilDefinition(expr, coefs, m, false)
end

"""
    CreateStencilDefinition(coefs::Array{Float64, 3}; uses_vsq=false)

Creates a Stencil definition struct given a coefficient array and whether
it should be multiplied by the vsq array (defaults to false.)
"""
function CreateStencilDefinition(coefs::Array{T, 3}; uses_vsq=false) where T <: Number
    # check coefs is cube

    coefs = Float64.(coefs)

    sz = size(coefs)
    !(sz[1] == sz[2] == sz[3]) &&  error("coefs array must have same dimensions size")
    max_radius = Int((sz[1] - 1)/2)

    return StencilDefinition(nothing, coefs, max_radius, uses_vsq)
end


"""
    function NewStencilInstance(varstencil::StencilDefinition...)


"""
function NewStencilInstance(varstencil::StencilDefinition...;prev_time_coeff=0, m_step=false, bdim=32)
    st_sym,max_radius = CombineStencils(varstencil...)
    fex,uses_vsq,fz,bz = GenStencilKernel(st_sym,max_radius,prev_time_coeff, bdim)
    kernel = ConvertToFunction(fex, uses_vsq)
    m_kernel = nothing
    m_rad = false
    m_fz = false
    m_bz = false
    fex2 = nothing
    new_sym = nothing
    if !uses_vsq && m_step != false && prev_time_coeff==0
        MAX_RAD = 16
        if max_radius*m_step > MAX_RAD
            m_step = false
            @warn "Combining timesteps results in too big stencil. Time combination aborted"
        else
            r = max_radius
            new_sym = combine_more_time_steps(st_sym, r, m_step)
            # for i = 2:m_step
            #     new_sym = CombineTimeSteps(st_sym, r, new_sym, r*i)
            # end
            m_rad = max_radius*m_step
            #println(new_sym)
            fex2,uses_vsq2,m_fz,m_bz = GenStencilKernel(new_sym,m_rad,prev_time_coeff, bdim)
            m_kernel = ConvertToFunction(fex2, uses_vsq2)
        end
    end
    return StencilInstance(varstencil, st_sym, max_radius,fz, bz, bdim, kernel, fex, uses_vsq,
                        m_step, m_kernel, fex2,new_sym, m_rad, m_fz, m_bz)
end

function ApplyStencil(st_inst::StencilInstance, org_data, t_steps::Integer; vsq=nothing)
    # Fix padding for multi time step
    #global timers
    to = TimerOutput()
    radius = st_inst.max_radius
    pad_radius = st_inst.max_radius
    offx = 0
    offy = 0
    if st_inst.combined_time_step != false
        pad_radius = st_inst.m_max_radius
        offx = st_inst.m_max_radius
        offy = st_inst.m_max_radius
    end

    # -------------------------------
    #radius = st_inst.max_radius
    data = PadData(pad_radius, org_data)
    bdimx = st_inst.bdim
    bdimy = st_inst.bdim
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
    bx = Int(floor((dx - 2*pad_radius)/bdimx))
    by = Int(floor((dy - 2*pad_radius)/bdimy))
    i = 0
    #st_inst.combined_time_step = false
    #println(st_inst.combined_time_step)
    @time begin
    while i < t_steps
        if st_inst.combined_time_step != false
            if i + st_inst.combined_time_step <= t_steps
                args = (dev_data, dev_out, 0, 0, 0, 0)
                if st_inst.uses_vsq
                    args = (args..., dev_vsq)
                end
                #println("Running combined $i")
                @timeit to "combined time step $(i)" begin
                @cuda(blocks=(bx,by,1), threads=(bdimx,bdimy),
                            shmem=((bdimx+2*st_inst.m_max_radius)*
                                (bdimy+2*st_inst.m_max_radius))*sizeof(Float32),
                            st_inst.m_kernel(args...))
                end # Timer end
                i += st_inst.combined_time_step
                dev_data,dev_out = dev_out,dev_data
                continue
            end
        end
        args = (dev_data, dev_out, 0, 0, offx, offy)
        if st_inst.uses_vsq
            args = (args..., dev_vsq)
        end
        #println("t = $(i), (bx,by) = $((bx,by))")

        @cuda(blocks=(bx,by,1), threads=(bdimx,bdimy),
                    shmem=((bdimx+2*radius)*(bdimy+2*radius))*sizeof(Float32),
                    st_inst.kernel(args...))

        dev_data,dev_out = dev_out,dev_data
        i += 1

    end # Timer end

    CUDA.synchronize(CuDefaultStream())
    end
    return view(Array(dev_data), (pad_radius+1):(dx-pad_radius), (pad_radius+1):(dy-pad_radius),:)

end

function ApplyFFTstencil(st_inst::StencilInstance, org_data, t_steps::Integer, multi=1)
    @show z = size(org_data, 3)
    template_single = matrix_from_expression(st_inst.stencil_sym, st_inst.max_radius)
    template_multi = nothing
    if multi > 1
        template_multi = DSP.conv(template_single, template_single)
        for i = 2:multi
            template_multi = DSP.conv(template_multi, template_single)
        end
    end

    ## 2d STENCIL
    if z == 1
        template_single = [:,:,st_inst.max_radius+1]
        if template_multi != nothing
            template_single = template_multi[:,:,multi*st_inst.max_radius+1]
        end
        org_data = org_data[:,:,1]
        step = 1
        if multi > 1
            step = multi
        end
        t_steps = t_steps÷step
        return fft_stencil_2d(org_data, template_single, t_steps)
    end
    if template_multi != nothing
        template_single = template_multi
    end
    step = 1
    if multi > 1
        step = multi
    end
    t_steps = t_steps÷step
    return fft_stencil_3d(org_data, template_single, t_steps)


end

# function ApplyStencil(st_inst::StencilInstance, data, t_steps::Integer; vsq=nothing)
#     radius = st_inst.max_radius
#     bdimx = 32
#     bdimy = 32
#     dx = size(data,1)
#     dy = size(data,2)
#     dev_data = CuArray(data)
#     dev_out = CUDA.zeros(Float32, size(data))
#     dev_vsq = nothing
#     if st_inst.uses_vsq
#         if vsq isa Nothing
#             error("Vsq array not provided")
#         end
#         dev_vsq = CuArray(vsq)
#     end
#     bx = Int(floor((dx - 2*radius)/bdimx))
#     by = Int(floor((dy - 2*radius)/bdimy))
#     for i = 1:t_steps
#         if i%2 == 1
#             if st_inst.uses_vsq
#                 @cuda(blocks=(bx,by,1), threads=(bdimx,bdimy),
#                             shmem=((bdimx+2*radius)*(bdimy+2*radius))*sizeof(Float32),
#                             st_inst.kernel(dev_data,dev_out,0,dev_vsq))
#             else
#                 @cuda(blocks=(bx,by,1), threads=(bdimx,bdimy),
#                             shmem=((bdimx+2*radius)*(bdimy+2*radius))*sizeof(Float32),
#                             st_inst.kernel(dev_data,dev_out,0))
#             end
#         else
#             if st_inst.uses_vsq
#                 @cuda(blocks=(bx,by,1), threads=(bdimx,bdimy),
#                             shmem=((bdimx+2*radius)*(bdimy+2*radius))*sizeof(Float32),
#                             st_inst.kernel(dev_out,dev_data,0,dev_vsq))
#             else
#                 @cuda(blocks=(bx,by,1), threads=(bdimx,bdimy),
#                             shmem=((bdimx+2*radius)*(bdimy+2*radius))*sizeof(Float32),
#                             st_inst.kernel(dev_out,dev_data,0))
#             end
#         end
#         println("t = $(i), (bx,by) = $((bx,by))")
#     end
#
#     (t_steps%2 == 1) ? (return Array(dev_out)) : (return Array(dev_data))
#
# end











#end
