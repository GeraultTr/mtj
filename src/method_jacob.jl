module MethodJacob

using QuadGK

const INT1_CACHE = Dict{NTuple{10, Float64}, Float64}()
const INT2_CACHE = Dict{NTuple{10, Float64}, Float64}()

pdist(x, X, σ) = (1 / sqrt(2 * π * σ)) * exp(-((x - X)^2) / (2 * σ))
attack(x, a, α, τ) = α * exp(-((x - a)^2) / (2 * τ^2))
handling(x, h, ηmin, ηmax, ν) = ηmax - (ηmax - ηmin) * exp(-((x - h)^2) / (2 * ν^2))

function _integration_limits(X, σ)
    if σ < 1
        return (X - 5, X + 5)
    else
        return (X - 5σ, X + 5σ)
    end
end

function int1(R, a, α, τ, h, ηmin, ηmax, ν, X, σ)
    key = (R, a, α, τ, h, ηmin, ηmax, ν, X, σ)
    if haskey(INT1_CACHE, key)
        return INT1_CACHE[key]
    end

    xmin, xmax = _integration_limits(X, σ)
    integrand(x) = (attack(x, a, α, τ) / (1 + attack(x, a, α, τ) * handling(x, h, ηmin, ηmax, ν) * R)) * pdist(x, X, σ)
    value, _ = quadgk(integrand, xmin, xmax; rtol=1e-6, atol=1e-6)
    INT1_CACHE[key] = value
    return value
end

function int2(R, a, α, τ, h, ηmin, ηmax, ν, X, σ)
    key = (R, a, α, τ, h, ηmin, ηmax, ν, X, σ)
    if haskey(INT2_CACHE, key)
        return INT2_CACHE[key]
    end

    xmin, xmax = _integration_limits(X, σ)
    integrand(x) = begin
        A = attack(x, a, α, τ)
        H = handling(x, h, ηmin, ηmax, ν)
        pd = pdist(x, X, σ)
        (A / (1 + A * H * R)) * pd * (1 - (A * H * R) / (1 + A * H * R))
    end
    value, _ = quadgk(integrand, xmin, xmax; rtol=1e-6, atol=1e-6)
    INT2_CACHE[key] = value
    return value
end

end # module
