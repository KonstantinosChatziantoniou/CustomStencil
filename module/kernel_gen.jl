using SyntaxTree
using SymEngine
using CUDA
include("types.jl")

function GenStencilKernel(st_sym, max_radius)

    function search_edges(stencil, max_radius, dim)
        radius = max_radius
        Dsym = [symbols("w_$(i)_$(j)_$(k)") for i = 1:(2*max_radius+1),
                       j = 1:(2*max_radius+1), k=1:(2*max_radius+1)]
        Doff = [[i,j,k] for i = -radius:radius, j=-radius:radius, k=-radius:radius]
        mn = 100
        mx = -100
        for i = 1:(2*max_radius+1), j = 1:(2*max_radius+1),k = 1:(2*max_radius+1)
            cf = SymEngine.coeff(stencil,Dsym[i,j,k])
            of = Doff[i,j,k]
            if cf != 0
                mn > of[dim] ? (mn = of[dim]) : nothing
                mx < of[dim] ? (mx = of[dim]) : nothing
            end
        end
        return (mn,mx)
    end
    bdim = 16
    mnx, mxx = search_edges(st_sym, max_radius, 1)
    mny, mxy = search_edges(st_sym, max_radius, 2)
    mnz, mxz = search_edges(st_sym, max_radius, 3)
    shmem_radius = max_radius#max( abs.((mnx, mxx, mny, mxy))... )
    frontz = mnz > 0 ? 0 : abs(mnz)
    behindz = mxz < 0 ? 0 : mxz
    q = CreateInitPart(shmem_radius, frontz, behindz, bdim)
    l,b1 = CreateLoopPart(shmem_radius, bdim)
    b2 = CreateCalcsPart(st_sym, shmem_radius, frontz, behindz)
    b3 = CreateRotateRegsPart(frontz, behindz)
    f = CreateFinalStorePart(shmem_radius, behindz)
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
    return linefilter!(q)
end


function ConvertToFunction(ex::Expr)

    return SyntaxTree.genfun(ex, [:g_input, :g_output])
end

function CreateInitPart(shmem_radius, frontz, behindz, bdim)
    radius = shmem_radius
    temp = bdim + 2*radius
    q = quote
        dimz = size(g_input,3)
        dimzend = dimz-$(radius)
        bdimx = $(CUDA).blockDim().x
        bdimy = $(CUDA).blockDim().y
        bx = ($(CUDA).blockIdx().x-1)*$(bdim) + 1
        by = ($(CUDA).blockIdx().y-1)*$(bdim) + 1
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
    for i = behindz:(-1):1
        sym = Symbol(string("behind", i))
        push!(q.args, :($(sym) = Float32(0.0)))
    end
    push!(q.args, :(current = Float32(0.0)))
    for i = 1:frontz
        sym = Symbol(string("infront", i))
        push!(q.args, :($(sym) = Float32(0.0)))
    end


    return q
end


function CreateCalcsPart(s, radius, frontz, behindz)
    max_radius = radius
    Dsym = [symbols("w_$(i)_$(j)_$(k)") for i = 1:(2*max_radius+1),
                   j = 1:(2*max_radius+1), k=1:(2*max_radius+1)]

    Doff = [[i,j,k] for i = -radius:radius, j=-radius:radius, k=-radius:radius]
    q = Expr(:block)
    for  k = radius+1,i = 1:size(Dsym,1), j = 1:size(Dsym,2)
        pos = Doff[i,j,k]
        sym = Dsym[i,j,k]
        cf = Float64(coeff(s, sym))
        if cf == 0
            continue
        end
        ### CASE FOR MAIN TILE XY
        ex = nothing
        if cf == 1
            ex = :(@inbounds current += tile[txr+$(pos[1]), tyr + $(pos[2])])
        else
            ex = :(@inbounds current += Float32($(cf))*tile[txr+$(pos[1]), tyr + $(pos[2])])
        end
        push!(q.args, ex)
    end

    for i = 1:frontz
        ## INFRONT
        sym = Symbol(string("infront",i))
        part_st     = @view Dsym[:,:,radius+1-i]
        part_ofst   = @view Doff[:,:,radius+1-i]
        for j = 1:(2*radius+1), k = 1:(2radius+1)
            st = part_st[j,k]
            of = part_ofst[j,k]
            cf = Float32(coeff(s, st))
            cf == 0 && continue
            if cf == 1
                push!(q.args, :(@inbounds $(sym) += tile[txr+$(of[1]), tyr+$(of[2])]))
            else
                push!(q.args, :(@inbounds $(sym) += Float32($(cf))*tile[txr+$(of[1]), tyr+$(of[2])]))
            end
        end
    end
    for i = 1:behindz
        ## BEHIND
        sym = Symbol(string("behind",i))
        part_st     = @view Dsym[:,:,radius+1+i]
        part_ofst   = @view Doff[:,:,radius+1+i]
        for j = 1:(2*radius+1), k = 1:(2radius+1)
            st = part_st[j,k]
            of = part_ofst[j,k]
            cf = Float32(coeff(s, st))
            cf == 0 && continue
            if cf == 1
                push!(q.args, :(@inbounds $(sym) += tile[txr+$(of[1]), tyr+$(of[2])]))
            else
                push!(q.args, :(@inbounds $(sym) += Float32($(cf))*tile[txr+$(of[1]), tyr+$(of[2])]))
            end
        end
    end
    if behindz == 0
        push!(q.args, :(if z > $(2*radius)
                            @inbounds l_output[txr,tyr,z-$(radius)] = current
                        end))
    else
        sym = Symbol(string("behind", behindz))
        push!(q.args, :(if z > $(2*radius)
                            @inbounds l_output[txr,tyr,z-$(radius)] = $(sym)
                        end))
    end
    return q
end

function CreateLoopPart(radius, bdim)
    l = Expr(:for)
    push!(l.args, :(z = $(radius+1):(dimzend)))

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

function CreateRotateRegsPart(frontz, behindz)
    regs = []

    for i = behindz:-1:1
        push!(regs, Symbol(string("behind", i)))
    end
    push!(regs, :current)
    for i = 1:frontz
        push!(regs, Symbol(string("infront",i )))
    end
    q = Expr(:block)
    for i = 1:(length(regs) - 1)
        push!(q.args, :($(regs[i]) = $(regs[i+1])))
    end
    sym = frontz == 0 ? :current : Symbol(string("infront", frontz))
    push!(q.args, :($(sym) = Float32(0)))
    return q
end

function CreateFinalStorePart(radius, behindz)
    q = Expr(:block)
    behindz == 0 && return :(return nothing)

    for i = 1:behindz
        sym = Symbol(string("behind", i))
        push!(q.args, :(@inbounds l_output[txr,tyr,dimz-$(radius + i-1)] = $(sym)))
    end

    push!(q.args, :(return nothing))
    return q
end
