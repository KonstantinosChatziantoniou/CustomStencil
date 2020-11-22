include("../../module/CustomStencil.jl")
include("../../misc/misc.jl")
include("../../misc/cpu_stencils.jl")
## Star Stencil definition with radius = 4



##

dense_coefs = zeros(9,9,9)
dense_coefs[1:9,1:9,5] .= 1
for i = parse(Int,ARGS[1])
    global dense_coefs
    for x = (i-1):-1:0
        for y = (i-1):-1:0
                dense_coefs[1+x,1+y,5] = 0
                dense_coefs[1+x,9-y,5] = 0
                dense_coefs[9-x,1+y,5] = 0
                dense_coefs[9-x,9-y,5] = 0
        end
    end
    @show sum(dense_coefs .== 1)/81
    st_def = CreateStencilDefinition(dense_coefs)
    bdim = 32
    if length(ARGS) == 2
        bdim = 16
    end
    st_inst1 = NewStencilInstance(st_def, bdim=bdim, m_step=false)
    println(st_inst1.kernel_expr)
    radius = 4
    nx = 12
    ny = 12
    nz = 0
    dx = 1<<(nx)
    dy = 1<<(ny)
    dz = 1<<(nz)

    data = CreateData(dx,dy,dz)

    t_steps = 16
    gpu_out = ApplyStencil(st_inst1, data, t_steps)
    CUDA.cuProfilerStart()
    NVTX.@range "standard" begin
        gpu_out = ApplyStencil(st_inst1, data, t_steps) end
    #exit()

end
