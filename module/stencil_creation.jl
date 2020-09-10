"""
    function CombineStencils(varstencil::StencilDefinition...)

Adds 2 or more stencils together, combines their coefficients.
Result has the radius of the bigger stencil.
Can only add stencils, substruction should be embedded in the stencil.

Returns:
    symbolic expression (Basic)
    max_radius          (Int)
"""
function CombineStencils(varstencil::StencilDefinition...)

    total_max_radius = -1
    for s in varstencil
        if s.max_radius > total_max_radius
            total_max_radius = s.max_radius
        end

    end

    Dsym = [symbols("w$(i)_$(j)_$(k)") for i = 1:(2total_max_radius+1),
                                        j = 1:(2total_max_radius+1),
                                        k = 1:(2total_max_radius+1)]
    global D = Dsym
    tot_s = 0
    for s in varstencil
        resss = convert_to_sym(s, total_max_radius)
        tot_s += resss

    end
    return expand(tot_s),total_max_radius
end



"""
function convert_to_sym(st::StencilDefinition, radius::Integer)

Converts a stencil expression, either array notation or coeff matrix, to
symbolic math. The extra radius parameter can be bigger than the defined
stencil's radius.
"""
function convert_to_sym(st::StencilDefinition, radius::Integer)
    if st.expr isa Nothing
        ## For coef stencil
        coefs = copy(st.coefs)
        if st.max_radius < radius
            pad = radius - st.max_radius
            coefs = zeros(Float32, 2*radius+1, 2*radius+1, 2*radius+1)

            for i = 1:(2st.max_radius+1), j = 1:(2st.max_radius+1), k = 1:(2st.max_radius+1)
                coefs[pad+i, pad+j,pad+k] = st.coefs[i,j,k]
            end
        end
        #println(size(coefs))
        global D
        st_sym = sum(coefs.*D)
        if st.uses_vsq
            global v
            st_sym *= v
        end
        return st_sym
    else
        ## For expression stencil
        coefs = nothing #copy(st.coefs[:])
        expr = copy(st.expr)
        for i in [:x :y :z]
            expr = replace_symbol(i, 0, expr)
        end
        if !(st.coefs isa Nothing)
            coefs = copy(st.coefs[:])
            if length(coefs) < radius
                pad = radius - length(coefs)
                for i = 1:pad
                    push!(coefs, 0)
                end
            end
            global c = [reverse(coefs[2:end]); coefs[:]]
        end

        expr = fix_indices(expr, radius+1)
        #println(expr)
        st_sym = eval(expr)
        return st_sym
    end
end

# function CheckTimestepCombinationLimit(limit_radius, radius, m_steps)
#     cr = radius
#     cm = 2
#     for i = 2:m_steps
#         cr = 2*cr
#         if cr > 16
#             cm = i-1
#             break
#         end
#     end
#
#     if cm < 2
#         return false
#     else
#         return cm
#     end
#
# end
function CreateSymOffMatrices(radius)
    Dsym = [symbols("w$(i)_$(j)_$(k)") for i = 1:(2radius+1),
                                        j = 1:(2radius+1),
                                        k = 1:(2radius+1)]

    Doff = [[i,j,k] for i = -radius:radius, j=-radius:radius, k=-radius:radius]

    return Dsym, Doff
end
function CreateOffset(radius)
    Doff = [[i,j,k] for i = -radius:radius, j=-radius:radius, k=-radius:radius]

    return Doff
end


function CombineTimeSteps(base_expr::Basic, base_radius::Int,
                            expr::Basic, radius::Int)


    b_sym, b_off = CreateSymOffMatrices(base_radius)
    sym,off = CreateSymOffMatrices(radius)
    new_sym, new_off = CreateSymOffMatrices(radius+base_radius)

    new_eq = 0

    for i = 1:(2*radius+1), j = 1:(2*radius+1),k = 1:(2*radius+1)
        c1 = coeff(expr, sym[i,j,k])
        (c1 == 0) && continue
        for ii = 1:(2*base_radius+1), jj = 1:(2*base_radius+1),kk = 1:(2*base_radius+1)
            c2 = coeff(base_expr, b_sym[ii,jj,kk])
            (c2 == 0) && continue
            off2 = b_off[ii,jj,kk] + [i;j;k] .+ base_radius
            new_eq += c1*c2*new_sym[off2...]
        end
    end
    return new_eq
end








function combine_more_time_steps(base_expr, base_radius, steps)
    steps = steps - 1
    final_radius = (1+steps)*base_radius
    all_syms,temp = CreateSymOffMatrices(final_radius)
    new_eq = 0
    b_sym = @view all_syms[1:2base_radius+1,1:2base_radius+1,1:2base_radius+1]
    b_off = CreateOffset(base_radius)
    expr = base_expr
    for t = 1:steps
        new_eq = 0
        radius = (t)*base_radius
        next_rad = (t+1)*base_radius
        off = CreateOffset(radius)
        sym = @view all_syms[1:2radius+1,1:2radius+1,1:2radius+1]
        new_off = CreateOffset(next_rad)
        new_sym = @view all_syms[1:2next_rad+1,1:2next_rad+1,1:2next_rad+1]
        for i = 1:(2*radius+1), j = 1:(2*radius+1),k = 1:(2*radius+1)
            c1 = coeff(expr, sym[i,j,k])
            (c1 == 0) && continue
            for ii = 1:(2*base_radius+1), jj = 1:(2*base_radius+1),kk = 1:(2*base_radius+1)
                c2 = coeff(base_expr, b_sym[ii,jj,kk])
                (c2 == 0) && continue
                off2 = b_off[ii,jj,kk] + [i;j;k] .+ base_radius
                new_eq += c1*c2*new_sym[off2...]
            end
        end
        expr = new_eq
    end
    return expr

end

function matrix_from_expression(expr, radius)
    sym,temp = CreateSymOffMatrices(radius)
    template = zeros(Float32, 2radius+1,2radius+1,2radius+1)
    for i = 1:(2*radius+1), j = 1:(2*radius+1),k = 1:(2*radius+1)
        c1 = coeff(expr, sym[i,j,k])
        (c1 == 0) && continue
        template[i,j,k] = c1
    end
    return template
end
"""
    function CombineTimeSteps(expr, radius)

Transform a stencil so it applies an additional timestep each execution.
"""
# function CombineTimeSteps(expr::Basic, radius::Integer)
#
#     Dsym = [symbols("w$(i)_$(j)_$(k)") for i = 1:(2radius+1),
#                                         j = 1:(2radius+1),
#                                         k = 1:(2radius+1)]
#
#     Doff = [[i,j,k] for i = -radius:radius, j=-radius:radius, k=-radius:radius]
#     new_Dsym = [symbols("w$(i)_$(j)_$(k)") for i = 1:(4radius+1),
#                                         j = 1:(4radius+1),
#                                         k = 1:(4radius+1)]
#     new_Doff =  [[i,j,k] for i = -2radius:2radius, j=-2radius:2radius, k=-2radius:2radius]
#
#     new_eq = 0
#     #@show 2*radius + 1
#     for i = 1:(2*radius+1), j = 1:(2*radius+1),k = 1:(2*radius+1)
#         #global d, Doff, Dsym, new_Dsym, new_Doff,new_eq,radius
#         #@show (i,j,k)
#         if coeff(expr,Dsym[i,j,k]) != 0
#             Dsym[i,j,k]
#             c1 = coeff(expr,Dsym[i,j,k])
#             for ii = 1:(2*radius+1), jj = 1:(2*radius+1),kk = 1:(2*radius+1)
#                 if coeff(expr,Dsym[ii,jj,kk]) != 0
#                     c = coeff(expr,Dsym[ii,jj,kk])
#                     off2 = Doff[ii,jj,kk] + [i;j;k] .+ radius
#                     new_eq += c1*c*new_Dsym[off2...]
#                 end
#             end
#         end
#
#     end
#     return new_eq
# end
