using SymEngine
function sabs(a)
    return abs.(a)
end
macro sum(i::Symbol,s::Integer ,e::Integer, expr::Expr)
    expr = eval_macro(expr, Symbol("@sum"))
    q = []
    for j = s:e
        push!(q,replace_symbol(i, j, expr))
    end
    r = Expr(:call)
    r.args = [:+; q]
    return Meta.quot(r)
end

function replace_symbol(i::Symbol, v::Integer, expr::Expr)
    temp = copy(expr)
    replace_symbol!(i, v, temp)
    return temp
end

function replace_symbol!(i::Symbol, v::Integer, expr::Expr)
    #println(typeof(expr))
    for (j,a) in enumerate(expr.args)
        if a == i
            expr.args[j] = v
        elseif a isa Expr
            replace_symbol!(i,v,a)
        end
    end
    return  expr
end

function eval_macro(expr::Expr, mc::Vararg{Symbol})
    temp = copy(expr)
    eval_macro!(temp, mc...)
    return temp
end

function eval_macro!(expr::Any, mc::Vararg{Symbol})
    if !(expr isa Expr)
        return
    end

    if expr.head == :macrocall
        if findfirst(expr.args[1] .== mc) != nothing
            x = eval(expr)
            expr.args = x.args
            expr.head = x.head
            return
        end
    end
    if expr isa Expr
        for (i, e) = enumerate(expr.args)
            eval_macro!(e, mc...)
        end
    end
    return expr
end
function eval_func!(expr::Any, mc::Vararg{Symbol})
    if !(expr isa Expr)
        return expr
    end

    if expr.head == :call
        if findfirst(expr.args[1] .== mc) != nothing
            x = eval(expr)
            return x
        end
    end
    if reduce( ==, expr.head .== mc)
        x = eval(expr)
        return x
    end
    if expr isa Expr
        for (i, e) = enumerate(expr.args)
            res = eval_func!(e, mc...)
            if res isa Number
                expr.args[i] =res
            end
        end
    end
    return expr
end

function minmax_const(expr::Expr)
    minval = 100
    maxval = -100
    function rec_minmax(expr, mn, mx, inref)
        for e in expr.args
            if e isa Expr
                (mn1, mx1) = rec_minmax(e, mn, mx,e.head == :ref || inref)
                if mn1 < mn
                    mn = mn1
                end
                if mx1 > mx
                    mx = mx1
                end
            elseif e isa Number
                !inref && continue
                flag = false
                if expr.head == :(-)
                    flag = true
                elseif expr.head == :(call)
                    if expr.args[1] == :(-)
                        flag = true
                    end
                end
                sgn = flag   ? -1 : 1
                if e*sgn < mn
                    mn = e*sgn
                end
                if e*sgn > mx
                    mx = e*sgn
                end
            end
        end
        return (mn,mx)
    end
    return rec_minmax(expr, minval, maxval, false)
end

function fix_indices(expr::Expr, offset::Integer)
    ex = copy(expr)
    function fix!(expr, offset, inref)
        for (i,e) in enumerate(expr.args)
            if e isa Expr
                 #println("rec", e)
                 fix!(expr.args[i], offset, e.head == :ref || inref)
            elseif e isa Number
                !inref && continue
                #println(e, " ", i, " ", expr.head, " ## " ,expr.args)
                if ((expr.head == :(call) && expr.args[1] == :(-)) ||
                                                expr.head == :(-))
                    expr.args[i] = -e + offset
                    if (expr.head == :(-))
                        expr.args[1] = :(+)
                    end
                    break
                elseif ((expr.head == :(call) && expr.args[1] == :(+)) ||
                    expr.head == :(+))
                    expr.args[i] = e + offset
                    break
                end
                expr.args[i] = e + offset
            end
        end
    end
    fix!(ex,offset, false)
    return ex
    #expr.head = (:call)
end

# macro def_stencil(expr::Expr)#, data, current, t)
#     expr = Meta.quot(expr)
#     expr = eval_macro(expr, Symbol("@sum"))
#     # mn,mx = minmax_const(expr)
#     # println("MIN", mn, " MAX ", mx)
#     # offset = 0
#     # if mn <= 0
#     #     offset = -mn + 1
#     # end
#     # println(offset," ", mn)
#     #e = fix_indices(expr, offset)
#     return expr
# end
#
# macro def_stencil(expr::Expr, vars, radius)
#     expr = Meta.quot(expr)
#     expr = eval_macro(expr, Symbol("@sum"))
#     stencil = expr
#     for i in vars.args
#         println(i.value)
#         stencil = replace_symbol(i.value, 0, stencil)
#     end
#     stencil = fix_indices(stencil, radius)
#     return (s=stencil, v=expr)
# end


nothing
