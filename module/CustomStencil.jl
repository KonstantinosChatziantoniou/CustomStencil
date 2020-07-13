module CustomStencil
include("types.jl")
include("parsing.jl")
include("kernel_gen.jl")


macro def_stencil_expression(expr::Expr)#, data, current, t)
    expr = Meta.quot(expr)
    expr = eval_macro(expr, Symbol("@sum"))
    for i in [:abs, :max, :min]
        expr = eval_func!(expr, i)
    end
    # mn,mx = minmax_const(expr)
    # println("MIN", mn, " MAX ", mx)
    # offset = 0
    # if mn <= 0
    #     offset = -mn + 1
    # end
    # println(offset," ", mn)
    #e = fix_indices(expr, offset)
    return expr
end

```
    Used by eval to parse an expression to symbolic math
```
D = Array{Basic,1}(undef,1)
c = Array{Float32,1}(undef,1)

#syms::Tuple{Symbol,Symbol,Symbol}
function NewStencilInstance(stencil::Expr, max_radius::Integer,
                         coefs::Array{Float64, M}) where M


    Dsyms = [symbols("w_$(i)_$(j)_$(k)") for i = 1:(2*max_radius+1),
                    j = 1:(2*max_radius+1), k=1:(2*max_radius+1)]
    stencil2 = copy(stencil)
    for i in [:x :y :z]
        stencil2 = replace_symbol(i, 0, stencil2)
    end
    stencil2 = fix_indices(stencil2, max_radius+1)
    global D = Dsyms
    global c = [reverse(coefs[2:end]); coefs[:]]
    st_sym = eval(stencil2)
    fex = GenStencilKernel(st_sym, max_radius)
    kernel = ConvertToFunction(fex)
    return StencilInstance(stencil, st_sym, max_radius, coefs, kernel, fex)
end

function NewStencilInstance(coefs::Array{Float64, 3})
    max_radius = Int((size(coefs,1)-1)/2)
    Dsyms = [symbols("w_$(i)_$(j)_$(k)") for i = 1:(2*max_radius+1),
                    j = 1:(2*max_radius+1), k=1:(2*max_radius+1)]
    st_sym = sum(coefs.*Dsyms)
    fex = GenStencilKernel(st_sym, max_radius)
    kernel = ConvertToFunction(fex)
    return  StencilInstance(Expr(:block), st_sym, max_radius, coefs, kernel, fex)
end
function ApplyStencil(st_inst::StencilInstance, data::Array{Float32,3}, t_steps::Integer)
    radius = st_inst.max_radius
    bdimx = 16
    bdimy = 16
    dx = size(data,1)
    dy = size(data,2)
    dev_data = CuArray(data)
    dev_out = CUDA.zeros(Float32, size(data))
    bx = Int(floor((dx - 2*radius)/bdimx))
    by = Int(floor((dy - 2*radius)/bdimy))
    for i = 1:t_steps
        if i%2 == 1
            @cuda(blocks=(bx,by,1), threads=(bdimx,bdimy),
                        shmem=((bdimx+2*radius)*(bdimy+2*radius))*sizeof(Float32),
                        st_inst.kernel(dev_data,dev_out))
        else
            @cuda(blocks=(bx,by,1), threads=(bdimx,bdimy),
                        shmem=((bdimx+2*radius)*(bdimy+2*radius))*sizeof(Float32),
                        st_inst.kernel(dev_out,dev_data))
        end
        println("t = $(i), (bx,by) = $((bx,by))")
    end

    (t_steps%2 == 1) ? (return Array(dev_out)) : (return Array(dev_data))

end


export NewStencilInstance
export ApplyStencil
export StencilInstance
export @def_stencil_expression
export @sum
export sabs




end
