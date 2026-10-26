module CAPSearch

using CAPReference
using RadiiPolynomial
using LinearAlgebra
using JSON3

const C = CAPReference
const REQUEST_SCHEMA = "cap.search.request/0.1"
const RESPONSE_SCHEMA = "cap.search.response/0.1"
const FRAME_LIMIT = 1 << 20

"""Immutable reference data. Each evaluation starts from these same coefficients
and inverse, never from the previous evaluation or an optimizer's warm start."""
struct Model
    problem::Dict{String,Any}
    candidate::Dict{String,Any}
    coefficients::Vector{Float64}
    inverse::Matrix{Float64}
end

function model(path)
    problem, _ = C.load_problem(path)
    candidate = C.candidate_document(path)
    coefficients = C.unhexfloat.(candidate["coefficients"])
    n = length(coefficients)
    inverse = [C.unhexfloat(candidate["inverse_rows"][i][j]) for i in 1:n, j in 1:n]
    return Model(problem, candidate, coefficients, inverse)
end

dimension(m::Model) = m.problem["id"] == "scalar-cuberoot" ? 1 : 2

function parameters(m::Model, x)
    C.require(x isa Vector && length(x) == dimension(m) &&
        all(v -> v isa Real && !(v isa Bool) && isfinite(v) && 0.5 <= v <= 1.5, x),
        "parameters must be finite inverse row scales in [0.5,1.5]")
    return Float64.(x)
end

function inverse(m::Model, values)
    x = parameters(m, values)
    return [m.inverse[i,j] * x[i == 1 ? 1 : 2] for i in axes(m.inverse,1), j in axes(m.inverse,2)]
end

"""Floating estimates only, evaluated at the fixed physical target epsilon.
No interval result or certification status is produced here. The actual
candidate must pass the unchanged CAPReference verifier and a fresh replay."""
function estimate(m::Model, values)
    A = inverse(m, values)
    epsilon = Float64(C.rational(m.problem["target_epsilon"]))
    if dimension(m) == 1
        u = m.coefficients[1]; a = A[1,1]
        Y = abs(a*(u^3-2.0))
        Z1 = max(abs(1.0-3.0*a*(u-epsilon)^2), abs(1.0-3.0*a*(u+epsilon)^2))
        Z2 = 0.0
    else
        k = 32
        s = evensym(Fourier(k, Float64(π)))
        u = Sequence(s, copy(m.coefficients))
        finite = LinearOperator(s,s,A)
        full = finite + C.PseudoInverseLaplacian()*(I-Projection(s))
        beta = 40.15625
        Y = norm(full*C.ch_f(u,0.0,beta),Ell1())
        projection = Projection(evensym(Fourier(3k+1,Float64(π))))
        finite_defect = opnorm(projection-full*(C.ch_df(u,0.0,beta)*projection),Ell1())
        tail = beta*(1.0+3.0*norm(u,Ell1())^2)/(Float64(π)*(k+2))^2
        Z1 = max(finite_defect,tail)
        normA = max(opnorm(finite,Ell1()),1.0/Float64(π)^2)
        Z2 = 3.0*beta*normA*(2.0*norm(u,Ell1())+epsilon)
    end
    polynomial = Y/epsilon + Z1-1.0 + 0.5*Z2*epsilon
    contraction = Z1+Z2*epsilon-1.0
    score = max(polynomial,contraction)
    C.require(all(isfinite, (Y,Z1,Z2,score)), "nonfinite floating estimate")
    return Dict("score" => score, "estimated_Y" => Y, "estimated_Z1" => Z1,
        "estimated_Z2" => Z2, "estimated_polynomial_normalized" => polynomial,
        "estimated_contraction_minus_one" => contraction, "physical_epsilon" => epsilon)
end

function candidate(m::Model, values)
    A = inverse(m,values)
    doc = deepcopy(m.candidate)
    # The approximation is still the reference Newton coefficient vector.
    # Its explicitly stored inverse is the untrusted, row-scaled proposal.
    doc["inverse_rows"] = [C.hexfloat.(A[i,:]) for i in axes(A,1)]
    return doc
end

"""Bound an incoming frame before allocating more than FRAME_LIMIT bytes."""
function read_frame(io; limit=FRAME_LIMIT)
    bytes = UInt8[]
    while !eof(io)
        b = read(io,UInt8)
        if b == UInt8('\n')
            return JSON3.read(String(bytes),Dict{String,Any})
        end
        C.require(length(bytes) < limit, "request frame exceeds limit")
        push!(bytes,b)
    end
    C.require(isempty(bytes), "truncated request frame")
    return nothing
end

function emit(io, value)
    data = JSON3.write(value)
    C.require(ncodeunits(data) <= FRAME_LIMIT, "response frame exceeds limit")
    println(io,data)
    flush(io)
end

function serve(problem_path; input=stdin, output=stdout)
    started = time_ns()
    m = model(problem_path)
    # Compile and populate immutable library caches before comparisons begin.
    nominal = ones(dimension(m))
    reference = estimate(m,nominal)
    emit(output,Dict("schema_version" => RESPONSE_SCHEMA,"request_id" => 0,"status" => "ready",
        "dimension" => dimension(m),"problem_id" => m.problem["id"],
        "problem_sha256" => m.candidate["problem_sha256"],"reference" => reference,
        "environment" => C.environment(),"server_sha256" => C.file_digest(joinpath(C.ROOT,"bin/search_server.jl")),
        "search_module_sha256" => C.file_digest(@__FILE__),"construction_seconds" => (time_ns()-started)/1e9))
    sequence = 0
    while true
        request = read_frame(input)
        request === nothing && return
        id = get(request,"request_id",nothing)
        C.require(id isa Integer && !(id isa Bool) && id == sequence+1, "request ID must increase by one")
        sequence = id
        op = get(request,"op",nothing)
        C.fields(request, op == "shutdown" ? ["schema_version","request_id","op"] :
            ["schema_version","request_id","op","parameters"])
        C.require(request["schema_version"] == REQUEST_SCHEMA, "request schema mismatch")
        if op == "shutdown"
            emit(output,Dict("schema_version" => RESPONSE_SCHEMA,"request_id" => id,"status" => "stopped"))
            return
        end
        C.require(op in ("evaluate","candidate"), "unsupported search operation")
        x = parameters(m,request["parameters"])
        if op == "evaluate"
            value = estimate(m,x)
            emit(output,Dict("schema_version" => RESPONSE_SCHEMA,"request_id" => id,
                "status" => "estimated","parameters" => x,"estimate" => value))
        else
            emit(output,Dict("schema_version" => RESPONSE_SCHEMA,"request_id" => id,
                "status" => "candidate","parameters" => x,"candidate" => candidate(m,x)))
        end
    end
end

end
