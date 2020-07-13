using SymEngine
mutable struct StencilInstance
    stencil::Expr
    stencil_sym::Basic
    max_radius::Int
    #syms::Tuple{Symbol,3}
    #data::Array{Float32,3}
    coefs::Array{Float32,N} where N
    kernel::Function
    kernel_expr::Expr
end
nothing
