function miscCombineTimeSteps(base_expr::Basic, base_radius::Int,
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


function countComp(expr, radius, m_step)
    count_arr = zeros(Int, m_step)
    r = radius
    new_sym = expr
    count_arr[1] = count_coeffs(expr, radius)
    for i = 2:m_step
        new_sym = CombineTimeSteps(expr, r, new_sym, r*i)
        count_arr[i] = count_coeffs(new_sym, r*i)
    end
    return count_arr
end

function count_coeffs(expr, radius)
    Dsym = [symbols("w$(i)_$(j)_$(k)") for i = 1:(2radius+1),
                                        j = 1:(2radius+1),
                                        k = 1:(2radius+1)]
    s = 0
    for d in Dsym
        c = coeff(expr, d)
        if c  != 0
            s += 1
        end
    end
    return s

end
