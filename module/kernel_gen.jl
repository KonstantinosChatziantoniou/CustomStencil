using SyntaxTree
using SymEngine
using CUDA
include("types.jl")

function GenStencilKernel(st_sym, max_radius,prev_time_coeff, bdim=32)

    function search_edges(stencil::Basic, max_radius::Integer, dim::Integer)
        radius = max_radius
        Dsym = [symbols("w$(i)_$(j)_$(k)") for i = 1:(2*max_radius+1),
                       j = 1:(2*max_radius+1), k=1:(2*max_radius+1)]
        Doff = [[i,j,k] for i = -radius:radius, j=-radius:radius, k=-radius:radius]
        mn = 100
        mx = -100
        for i = 1:(2*max_radius+1), j = 1:(2*max_radius+1),k = 1:(2*max_radius+1)
            cf = SymEngine.coeff(stencil,Dsym[i,j,k])
            of = Doff[i,j,k]
            if cf != 0
                mn > of[dim] ? (mn= of[dim]) : nothing
                mx < of[dim] ? (mx= of[dim]) : nothing
            end
        end
        return (mn,mx)
    end
    #bdim = 32
    mnx, mxx = search_edges(st_sym, max_radius, 1)
    mny, mxy = search_edges(st_sym, max_radius, 2)
    mnz, mxz = search_edges(st_sym, max_radius, 3)
    shmem_radius = max_radius#max( abs.((mnx, mxx, mny, mxy))... )
    v_info, nv_info = GenerateStencil(st_sym, shmem_radius)
    q = CreateInitPart(shmem_radius, bdim, v_info, nv_info)
    l,b1 = CreateLoopPart(shmem_radius, bdim)
    b2 = CreateCalcsPart(v_info, nv_info, shmem_radius,prev_time_coeff=prev_time_coeff)
    b3 = CreateRotateRegsPart(v_info, nv_info)
    f = CreateFinalStorePart(v_info, nv_info,prev_time_coeff=prev_time_coeff)
    for i in b2.args
        push!(b1.args, i)
    end
    for i in b3.args
        push!(b1.args, i)
    end
    push!(l.args, b1)
    push!(q.args, l)
    for i in f.args
        push!(q.args,i)
    end
    return linefilter!(q), v_info.exists, abs(mxz), abs(mnz)
end



"""
    search_edges(stencil, max_radius, dim)

    For a `stencil`'s symbolic math expression, find how much it extends
    to the given dimesnion `dim`. `max_radius` is used to create the array of
    math symbols.

    Returns the minimum and maximum index
"""
function search_edges(stencil::Basic, max_radius::Integer, dim::Integer)
    radius = max_radius
    Dsym = [symbols("w$(i)_$(j)_$(k)") for i = 1:(2*max_radius+1),
                   j = 1:(2*max_radius+1), k=1:(2*max_radius+1)]
    Doff = [[i,j,k] for i = -radius:radius, j=-radius:radius, k=-radius:radius]
    mn = 100
    mx = -100
    for i = 1:(2*max_radius+1), j = 1:(2*max_radius+1),k = 1:(2*max_radius+1)
        cf = SymEngine.coeff(stencil,Dsym[i,j,k])
        of = Doff[i,j,k]
        # if dim == 3 && cf != 0
        #     @show cf, of
        # end
        if cf != 0
            mn > of[dim] ? (mn= of[dim]) : nothing
            mx < of[dim] ? (mx= of[dim]) : nothing
        end
    end
    return (mn,mx)
end


function GenerateStencil(st_sym::Basic, radius::Integer)
    ## Check if it uses vsq array
    check_vsq = coeff(st_sym, v) != 0
    vsq_stencil = nothing
    non_vsq_stencil = st_sym
    ## Split vsq stencil and non vsq stencil
    if check_vsq
        vsq_stencil = coeff(st_sym,v)
        non_vsq_stencil = expand(st_sym - v*vsq_stencil)
    end
    ## ----------------------------------------
    #           FIND INFO FOR EACH STENCIL
    ## ----------------------------------------
    vsq_info = stencil_info()
    if !check_vsq
        vsq_info.exists = false
    else
        vsq_info.exists = true
        vsq_info.sym_math = vsq_stencil
        vsq_info.max_radius = radius
        mnz, mxz = search_edges(vsq_stencil, radius, 3)
        # front cannot be negative
        vsq_info.front_max = mnz > 0 ? 0 : abs(mnz)
        # behind can't be positive
        vsq_info.behind_max = mxz < 0 ? 0 : mxz
    end
    non_vsq_info = stencil_info()
    if non_vsq_stencil == 0
        non_vsq_info.exists = false
    else
        non_vsq_info.exists = true
        non_vsq_info.sym_math = non_vsq_stencil
        non_vsq_info.max_radius = radius
        mnz, mxz = search_edges(non_vsq_stencil, radius, 3)
        # front cannot be negative
        non_vsq_info.front_max = mnz > 0 ? 0 : abs(mnz)
        # behind can't be positive
        non_vsq_info.behind_max = mxz < 0 ? 0 : mxz
    end

    ## FIX front/behind_max to match both stencils
    if vsq_info.exists == non_vsq_info.exists == true
        # data have to be passed from front to behind
        # until they are stored. So both stencil must
        # have the same amount of behind registers.
        if non_vsq_info.behind_max > vsq_info.behind_max
            vsq_info.behind_max = non_vsq_info.behind_max
        else
            non_vsq_info.behind_max = vsq_info.behind_max
        end
    end
    return (vsq_info, non_vsq_info)
end


# function convert_to_sym(st::StencilDefinition, radius::Integer)
#     if st.expr isa Nothing
#         ## For coef stencil
#         coefs = copy(st.coefs)
#         if st.max_radius < radius
#             pad = radius - st.max_radius
#             coefs = zeros(Float64, 2*radius+1, 2*radius+1, 2*radius+1)
#             # for i in  length(coefs)
#             #     coefs[i] = (0)
#             #     println(coefs)
#             # end
#             # println(coefs)
#             #println(size(coefs))
#             for i = 1:(2st.max_radius+1), j = 1:(2st.max_radius+1), k = 1:(2st.max_radius+1)
#                 coefs[pad+i, pad+j,pad+k] = st.coefs[i,j,k]
#                 #println(size(coefs))
#             end
#         end
#         #println(size(coefs))
#         global D
#         D = [symbols("w$(i)_$(j)_$(k)") for i = 1:(2radius+1),
#                                             j = 1:(2radius+1),
#                                             k = 1:(2radius+1)]
#         st_sym = sum(coefs.*D)
#         if st.usus_vsq
#             global v
#             st_sym *= v
#         end
#         return st_sym
#     else
#         ## For expression stencil
#         coefs = copy(st.coefs[:])
#         expr = copy(st.expr)
#         for i in [:x :y :z]
#             expr = replace_symbol(i, 0, expr)
#         end
#         if length(coefs) < radius
#             pad = radius - length(coefs)
#             for i = 1:pad
#                 push!(coefs, 0)
#             end
#         end
#         global c = [reverse(coefs[2:end]); coefs[:]]
#         expr = fix_indices(expr, radius+1)
#         st_sym = eval(expr)
#         return st_sym
#     end
# end

function ConvertToFunction(ex::Expr, uses_vsq::Bool)
    if uses_vsq
        return SyntaxTree.genfun(ex, [:g_input, :g_output, :offz_f, :offz_b, :offx, :offy, :g_vsq])
    else
        return SyntaxTree.genfun(ex, [:g_input, :g_output, :offz_f, :offz_b, :offx, :offy])
    end
end

function CreateInitPart(shmem_radius::Integer, bdim::Integer,
                vsq_st::stencil_info, non_vsq_st::stencil_info)
    radius = shmem_radius
    temp = bdim + 2*radius
    q = quote
        dimz = size(g_input,3)
        dimzend = dimz-$(radius)
        bdimx = $(CUDA).blockDim().x
        bdimy = $(CUDA).blockDim().y
        bx = ($(CUDA).blockIdx().x-1)*$(bdim) + 1 + offx
        by = ($(CUDA).blockIdx().y-1)*$(bdim) + 1 + offy
        tx = $(CUDA).threadIdx().x
        ty = $(CUDA).threadIdx().y
        txr = $(CUDA).threadIdx().x+$(radius)
        tyr = $(CUDA).threadIdx().y+$(radius)
        tile = $(CUDA).@cuDynamicSharedMem(Float32,
                                   ($(temp), $(temp)))

        @inbounds l_input = @view g_input[bx:(bx+$(temp-1)),
                               by:(by+$(temp-1)), :]
        @inbounds l_output = @view g_output[bx:(bx+$(temp-1)),
                               by:(by+$(temp-1)), :]
    end
    if non_vsq_st.exists
        for i = non_vsq_st.behind_max:(-1):1
            sym = Symbol(string("behind", i))
            push!(q.args, :($(sym) = Float32(0.0)))
        end
        push!(q.args, :(current = Float32(0.0)))
        for i = 1:non_vsq_st.front_max
            sym = Symbol(string("infront", i))
            push!(q.args, :($(sym) = Float32(0.0)))
        end
    end
    """Add the init for non vsq registers
        vsq_infront max is at least current
        vsq_behind max is as normal begind max
    """
    if vsq_st.exists
        push!(q.args, :(@inbounds l_vsq = @view g_vsq[bx:(bx+$(bdim-1)),
                                   by:(by+$(bdim-1)), :]))
        for i = vsq_st.behind_max:(-1):1
            sym = Symbol(string("vsq_behind", i))
            push!(q.args, :($(sym) = Float32(0.0)))
        end
        push!(q.args, :(vsq_current = Float32(0.0)))
        for i = 1:vsq_st.front_max
            sym = Symbol(string("vsq_infront", i))
            push!(q.args, :($(sym) = Float32(0.0)))
        end
    end
    return linefilter!(q)
end

function CreateCalcsPart(vsq_st, non_vsq_st, radius; prev_time_coeff=0)
    max_radius = radius
    Dsym = [symbols("w$(i)_$(j)_$(k)") for i = 1:(2*max_radius+1),
                   j = 1:(2*max_radius+1), k=1:(2*max_radius+1)]

    Doff = [[i,j,k] for i = -radius:radius, j=-radius:radius, k=-radius:radius]
    q = Expr(:block)
    # ## Determine Calculation for the 2d xy Tile
    # for  k = radius+1,i = 1:size(Dsym,1), j = 1:size(Dsym,2)
    #     pos = Doff[i,j,k]
    #     sym = Dsym[i,j,k]
    #     if vsq_st.exists
    #         cf = Float64(coeff(vsq_st.sym_math, sym))
    #         if cf != 0
    #             ex = :()
    #             if cf == 1
    #                 ex = :(@inbounds vsq_current += tile[txr+$(pos[1]), tyr + $(pos[2])])
    #             else
    #                 ex = :(@inbounds vsq_current += Float32($(cf))*tile[txr+$(pos[1]), tyr + $(pos[2])])
    #             end
    #             push!(q.args, ex)
    #         end
    #     end
    #     if non_vsq_st.exists
    #         cf = Float64(coeff(non_vsq_st.sym_math, sym))
    #         if cf != 0
    #             ex = :()
    #             if cf == 1
    #                 ex = :(@inbounds current += tile[txr+$(pos[1]), tyr + $(pos[2])])
    #             else
    #                 ex = :(@inbounds current += Float32($(cf))*tile[txr+$(pos[1]), tyr + $(pos[2])])
    #             end
    #             push!(q.args, ex)
    #         end
    #     end
    # end

    ## Determine Calculation for contribution at infront and behind
    for x = 1:(2*max_radius+1), y = 1:(2*max_radius+1)
        of = Doff[x,y,1]
        read_to_temp = :(@inbounds temp = tile[txr+$(of[1]), tyr+$(of[2])])
        flag_read_temp = true

        # Current
        pos = Doff[x,y,max_radius+1]
        sym = Dsym[x,y,max_radius+1]
        if vsq_st.exists
            cf = Float64(coeff(vsq_st.sym_math, sym))
            if cf != 0
                if flag_read_temp
                    push!(q.args, read_to_temp)
                    flag_read_temp = false
                end
                ex = :()
                if cf == 1
                    ex = :(vsq_current += temp)
                else
                    ex = :(vsq_current += Float32($(cf))*temp)
                end
                push!(q.args, ex)
            end
        end
        if non_vsq_st.exists
            cf = Float64(coeff(non_vsq_st.sym_math, sym))
            if cf != 0
                if flag_read_temp
                    push!(q.args, read_to_temp)
                    flag_read_temp = false
                end
                ex = :()
                if cf == 1
                    ex = :(current += temp)
                else
                    ex = :(current += Float32($(cf))*temp)
                end
                push!(q.args, ex)
            end
        end





        for z = 1:max_radius
        ## FRONT
            var = Dsym[x,y,z]
            off = Doff[x,y,z]
            if vsq_st.exists
                sym = Symbol(string("vsq_infront",abs(off[3])))
                cf = Float64(coeff(vsq_st.sym_math, var))
                if cf != 0
                    if flag_read_temp
                        push!(q.args, read_to_temp)
                        flag_read_temp = false
                    end
                    if cf == 1
                        push!(q.args, :($(sym) += temp))
                    else
                        push!(q.args, :($(sym) += Float32($(cf))*temp))
                    end
                end
            end

            if non_vsq_st.exists
                sym = Symbol(string("infront",abs(off[3])))
                cf = Float64(coeff(non_vsq_st.sym_math, var))
                if cf != 0
                    if flag_read_temp
                        push!(q.args, read_to_temp)
                        flag_read_temp = false
                    end
                    if cf == 1
                        push!(q.args, :($(sym) += temp))
                    else
                        push!(q.args, :($(sym) += Float32($(cf))*temp))
                    end
                end
            end
        end
        for z = 1:max_radius
        ## Behind
            var = Dsym[x,y,2*max_radius + 2 - z]
            off = Doff[x,y,2*max_radius + 2 - z]
            if vsq_st.exists
                sym = Symbol(string("vsq_behind",abs(off[3])))
                cf = Float64(coeff(vsq_st.sym_math, var))
                if cf != 0
                    if flag_read_temp
                        push!(q.args, read_to_temp)
                        flag_read_temp = false
                    end
                    if cf == 1
                        push!(q.args, :($(sym) += temp))
                    else
                        push!(q.args, :($(sym) += Float32($(cf))*temp))
                    end
                end
            end

            if non_vsq_st.exists
                sym = Symbol(string("behind",abs(off[3])))
                cf = Float64(coeff(non_vsq_st.sym_math, var))
                if cf != 0
                    if flag_read_temp
                        push!(q.args, read_to_temp)
                        flag_read_temp = false
                    end
                    if cf == 1
                        push!(q.args, :($(sym) += temp))
                    else
                        push!(q.args, :($(sym) += Float32($(cf))*temp))
                    end
                end
            end
        end
    end

    ## SAVE RESULTS
    if non_vsq_st.exists == vsq_st.exists == true
        if non_vsq_st.behind_max == 0
            if prev_time_coeff == 0
                push!(q.args, :(@inbounds l_output[txr,tyr,z] =
                                         vsq_current*l_vsq[tx,ty,z] + current))
            else
                push!(q.args, :(@inbounds l_output[txr,tyr,z] =
                                    vsq_current*l_vsq[tx,ty,z] +
                                    current + Float32($(prev_time_coeff))*l_output[txr,tyr,z]))
            end
        else
            bmax = non_vsq_st.behind_max
            snv = Symbol(string("behind", non_vsq_st.behind_max))
            sv = Symbol(string("vsq_behind", non_vsq_st.behind_max))
            if prev_time_coeff == 0
                push!(q.args, :(if z > $(bmax)
                                @inbounds l_output[txr,tyr,z-$(bmax)] = $(sv)*l_vsq[tx,ty,z-$(bmax)] + $(snv)
                            end))
            else
                push!(q.args, :(if z > $(bmax)
                                @inbounds l_output[txr,tyr,z-$(bmax)] = $(sv)*l_vsq[tx,ty,z-$(bmax)] + $(snv) + Float32($(prev_time_coeff)*l_output[txr,tyr,z-$(bmax)])
                            end))
            end

        end

    else
        if non_vsq_st.exists
            if non_vsq_st.behind_max == 0
                if prev_time_coeff == 0
                    push!(q.args, :(
                                    @inbounds l_output[txr,tyr,z] = current))
                else
                    push!(q.args, :(
                                    @inbounds l_output[txr,tyr,z] = current + Float32($(prev_time_coeff))*l_output[txr,tyr,z]
                                ))
                end
            else
                bmax = non_vsq_st.behind_max
                sym = Symbol(string("behind", non_vsq_st.behind_max))
                if prev_time_coeff == 0
                    push!(q.args, :(if z > $(bmax)
                                    @inbounds l_output[txr,tyr,z-$(bmax)] = $(sym)
                                end))
                else
                    push!(q.args, :(if z > $(bmax)
                                    @inbounds l_output[txr,tyr,z-$(bmax)] = $(sym) + Float32($(prev_time_coeff))*l_output[txr,tyr,z-$(bmax)]
                                end))

                end
            end
        end

        if vsq_st.exists
            if vsq_st.behind_max == 0
                if prev_time_coeff == 0
                    push!(q.args, :(
                                    @inbounds l_output[txr,tyr,z] = vsq_current
                                ))
                else
                    push!(q.args, :(
                                    @inbounds l_output[txr,tyr,z] = vsq_current + Float32($(prev_time_coeff))*l_output[txr,tyr,z]
                                ))
                end
            else
                bmax = vsq_st.behind_max
                sym = Symbol(string("vsq_behind", vsq_st.behind_max))
                if prev_time_coeff == 0
                    push!(q.args, :(if z > $(bmax)
                                    @inbounds l_output[txr,tyr,z-$(bmax)] = $(sym)*l_vsq[tx,ty,z-$(bmax)]
                                end))
                else
                    push!(q.args, :(if z > $(bmax)
                                    @inbounds l_output[txr,tyr,z-$(bmax)] = $(sym)*l_vsq[tx,ty,z-$(bmax)] + Float32($(prev_time_coeff))*l_output[txr,tyr,z-$(bmax)]
                                end))
                end
            end
        end



    end
    return linefilter!(q)
end

function CreateLoopPart(radius, bdim)
    #radius = max(non_vsq_st.behind_max, vsq_st.behind_max)
    l = Expr(:for)
    push!(l.args, :(z = ((1+offz_f)):((dimz-offz_b))))

    q = Expr(:block)
    push!(q.args, :($(CUDA).sync_threads()))

    len = bdim + 2*radius
    for i = 0:bdim:len
        for j = 0:bdim:len
            tq = :(
                 if tx+$(i) <= $(len) && ty+$(j) <= $(len)
                     @inbounds tile[tx+$(i), ty+$(j)] = l_input[tx+$(i), ty+$(j),z]
                 end
            )

            push!(q.args, tq)
        end
    end
    push!(q.args, :(($CUDA).sync_threads()))
    #push!(l.args, q)
    return l,q
end

function CreateRotateRegsPart(vsq_st, non_vsq_st)
    q = Expr(:block)
    if non_vsq_st.exists
        regs = []

        for i = non_vsq_st.behind_max:-1:1
            push!(regs, Symbol(string("behind", i)))
        end
        push!(regs, :current)
        for i = 1:non_vsq_st.front_max
            push!(regs, Symbol(string("infront",i )))
        end
        for i = 1:(length(regs) - 1)
            push!(q.args, :($(regs[i]) = $(regs[i+1])))
        end
        sym = non_vsq_st.front_max == 0 ? :current : Symbol(string("infront", non_vsq_st.front_max))
        push!(q.args, :($(sym) = Float32(0)))
    end
    if vsq_st.exists
        regs = []

        for i = vsq_st.behind_max:-1:1
            push!(regs, Symbol(string("vsq_behind", i)))
        end
        push!(regs, :vsq_current)
        for i = 1:vsq_st.front_max
            push!(regs, Symbol(string("vsq_infront",i )))
        end
        for i = 1:(length(regs) - 1)
            push!(q.args, :($(regs[i]) = $(regs[i+1])))
        end
        sym = vsq_st.front_max == 0 ? :vsq_current : Symbol(string("vsq_infront", vsq_st.front_max))
        push!(q.args, :($(sym) = Float32(0)))
    end
    return q
end

function CreateFinalStorePart(vsq_st, non_vsq_st; prev_time_coeff = 0)
    q = Expr(:block)
    behindz = max(vsq_st.behind_max, non_vsq_st.behind_max)
    behindz == 0 && return :(return nothing)

    for i = 1:behindz
        ## SAVE RESULTS
        if non_vsq_st.exists == vsq_st.exists == true
            bmax = i
            snv = Symbol(string("behind", i))
            sv = Symbol(string("vsq_behind", i))
            if prev_time_coeff == 0
                push!(q.args, :(@inbounds l_output[txr,tyr,dimz-$(bmax-1)-offz_b] = $(sv)*l_vsq[tx,ty,dimz-$(bmax-1)-offz_b] + $(snv)))
            else
                push!(q.args, :(@inbounds l_output[txr,tyr,dimz-$(bmax-1)-offz_b] = $(sv)*l_vsq[tx,ty,dimz-$(bmax-1)-offz_b] + $(snv) + Float32($(prev_time_coeff))*l_output[txr,tyr,dimz-$(bmax-1)-offz_b]))
            end
        else
            if non_vsq_st.exists
                bmax = i#non_vsq_st.behind_max
                sym = Symbol(string("behind", i))
                if prev_time_coeff == 0
                    push!(q.args, :(@inbounds l_output[txr,tyr,dimz-$(bmax-1)-offz_b] = $(sym)))
                else
                    push!(q.args, :(@inbounds l_output[txr,tyr,dimz-$(bmax-1)-offz_b] = $(sym) + Float32($(prev_time_coeff)*l_output[txr,tyr,dimz-$(bmax-1)-offz_b])))

                end
            end

            if vsq_st.exists
                bmax = i#non_vsq_st.behind_max
                sym = Symbol(string("vsq_behind", i))
                if prev_time_coeff == 0
                    push!(q.args, :(@inbounds l_output[txr,tyr,dimz-$(bmax-1)-offz_b] = $(sym)*l_vsq[tx,ty,dimz-$(bmax-1)-offz_b]))
                else
                    push!(q.args, :(@inbounds l_output[txr,tyr,dimz-$(bmax-1)-offz_b] = $(sym)*l_vsq[tx,ty,dimz-$(bmax-1)-offz_b] + Float32($(prev_time_coeff))*l_output[txr,tyr,dimz-$(bmax-1)-offz_b]))
                end
            end



        end
    end

    push!(q.args, :(return nothing))
    return q
end
