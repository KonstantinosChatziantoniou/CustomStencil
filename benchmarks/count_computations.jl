include("../module/CustomStencil.jl")
include("../misc/misc.jl")
include("../misc/comp_count.jl")
include("../misc/cpu_stencils.jl")

## Star Stencil definition with radius = 1
coefs = [1;0.5;]
star_stencil = @def_stencil_expression c[0]D[x,y,z] + @sum(i, 1, 1, c[i]*(
            D[x+i,y,z] + D[x,y+i,z] + D[x,y,z+i] +
            D[x-i,y,z] + D[x,y-i,z] + D[x,y,z-i]))
st_def = CreateStencilDefinition(star_stencil, coefs)
st_single = NewStencilInstance(st_def, m_step=false)
sym = st_single.stencil_sym
countComp(sym,1,7)
## Dense Stencil definition with radius = 1
coefs = [0.5;0.25]
dense_stencil = @def_stencil_expression(
            @sum(i,-1,1,
                @sum(j,-1,1,
                    @sum(k,-1,1, c[max(abs(i),abs(j),abs(k))]*D[x+i,y+j,z+k]))))

st_def = CreateStencilDefinition(dense_stencil, coefs)
st_single = NewStencilInstance(st_def, m_step=false)
sym = st_single.stencil_sym
countComp(sym,1,4)
## Flux summation Stencil definition with radius = 1
coefs = [1;0.2]
flux_stencil = @def_stencil_expression c[0]D[x,y,z] + @sum(i, 1,1, c[i]*(
            D[x+i,y,z] + D[x,y+i,z] + D[x,y,z+i]))
st_def = CreateStencilDefinition(flux_stencil, coefs)
st_single = NewStencilInstance(st_def, m_step=false)
sym = st_single.stencil_sym
a = countComp(sym,1,8)
