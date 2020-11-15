include("../../module/CustomStencil.jl")
include("../../misc/misc.jl")
include("../../misc/cpu_stencils.jl")
## Star Stencil definition with radius = 4



##

dense_coefs = ones(9,9,9)

for i = parse(Int, ARGS[1])
    global dense_coefs
    for x = (i-1):-1:0
        for y = (i-1):-1:0
            for z = (i-1):-1:0
                dense_coefs[1+x,1+y,1+z] = 0
                dense_coefs[1+x,1+y,9-z] = 0
                dense_coefs[1+x,9-y,1+z] = 0
                dense_coefs[1+x,9-y,9-z] = 0
                dense_coefs[9-x,1+y,1+z] = 0
                dense_coefs[9-x,1+y,9-z] = 0
                dense_coefs[9-x,9-y,1+z] = 0
                dense_coefs[9-x,9-y,9-z] = 0
            end
        end
    end
    st_def = CreateStencilDefinition(dense_coefs)
    st_inst1 = NewStencilInstance(st_def, m_step=false)

    radius = 4
    nx = 8
    ny = 8
    nz = 8
    dx = 1<<(nx)
    dy = 1<<(ny)
    dz = 1<<(nz)

    data = CreateData(dx,dy,dz)

    t_steps = 16
    gpu_out = ApplyFFTstencil(st_inst1, data, t_steps)
    CUDA.cuProfilerStart()
    NVTX.@range "r1" begin
        gpu_out = ApplyFFTstencil(st_inst1, data, t_steps,1) end
    NVTX.@range "r2" begin
        gpu_out = ApplyFFTstencil(st_inst1, data, t_steps,2) end
    NVTX.@range "r3" begin
        gpu_out = ApplyFFTstencil(st_inst1, data, t_steps,4) end
    NVTX.@range "r4" begin
        gpu_out = ApplyFFTstencil(st_inst1, data, t_steps,8) end
    NVTX.@range "r5" begin
        gpu_out = ApplyFFTstencil(st_inst1, data, t_steps,16) end
    exit()

end
