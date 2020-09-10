using SymEngine

"""
Saves info given by the user for a given stencil.
There are 2 "modes":
    + Expression with optional coefficients.
    + A cubic matrix with coefficient and uses_vsq flag.
"""
mutable struct StencilDefinition
    expr::Union{Expr, Nothing}
    coefs::Union{Array{Float64, N} where N ,Array{Basic, N} where N ,Nothing}
    max_radius::Integer
    uses_vsq::Bool
end


"""
Saves the generated function and other revelant info.
"""
mutable struct StencilInstance
    stencil::Union{StencilDefinition, NTuple{N,StencilDefinition} where N}
    stencil_sym::Basic
    max_radius::Int
    front_z_max::Int
    behind_z_max::Int
    bdim::Int
    #syms::Tuple{Symbol,3}
    #data::Array{Float32,3}
    #coefs::Array{Float32,N} where N
    kernel::Function
    kernel_expr::Expr
    uses_vsq::Bool
    combined_time_step::Union{Bool, Int}
    m_kernel::Union{Nothing, Function}
    m_kernel_exp::Union{Nothing, Expr}
    m_kernel_sym::Union{Nothing, Basic}
    m_max_radius::Union{Bool, Int}
    m_front_z_max::Union{Bool, Int}
    m_behind_z_max::Union{Bool, Int}
end


mutable struct stencil_info
    exists::Bool
    sym_math::Basic
    front_max::Integer
    behind_max::Integer
    max_radius::Integer
    stencil_info() = new(false, symbols(:a), 0,0,0)
end
nothing
