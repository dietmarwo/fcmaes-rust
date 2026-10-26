module CAPReference

using RadiiPolynomial
using IntervalArithmetic
using LinearAlgebra
using JSON3
using SHA
using TOML

const ROOT = normpath(joinpath(@__DIR__, ".."))
const MAX_DOCUMENT_BYTES = 1 << 20
const RP_COMMIT = "704009d7aa2e621c0900c9c158e853cb84c0c253"
const IA_COMMIT = "56701d7324292eb0de5edfe639bfa59d2e7b19b3"
const ENCODING = "ieee754-binary64-hex"
const ENV_VERSIONS = ["julia", "radii_polynomial", "interval_arithmetic"]
const ENV_COMMITS = ["radii_commit", "interval_commit"]
const ENV_HASHES = ["project_sha256", "manifest_sha256", "verifier_sha256", "cli_sha256"]
const ENV_THREADS = ["julia_threads", "blas_threads"]
# Interval and convolution settings the verifier runs with. A startup file or a
# caller can change them, for example to round-to-nearest, while every result
# still reports itself as guaranteed; the generic interval matrix product keeps
# enclosures independent of the CPU's BLAS kernels.
const ARITHMETIC = "Float64 set_based correct-rounding fast-power slow-matmul loop-convolution"
const IA = IntervalArithmetic

require(ok, message) = ok ? nothing : throw(ArgumentError(message))
digest(bytes) = bytes2hex(sha256(bytes))
file_digest(path) = open(sha256, path) |> bytes2hex

function fields(value, names)
    require(value isa AbstractDict && Set(keys(value)) == Set(names),
        "unexpected or missing fields; expected $(join(names, ", "))")
    return value
end

function metadata(doc)
    value = doc["elapsed_seconds"]
    require(value isa Real && !(value isa Bool) && isfinite(value) && value >= 0, "invalid elapsed time")
    env = fields(doc["environment"], [ENV_VERSIONS; ENV_COMMITS; ENV_HASHES; ENV_THREADS; "arithmetic"])
    require(env["arithmetic"] == ARITHMETIC, "invalid arithmetic metadata")
    for key in ENV_VERSIONS
        require(env[key] isa String && occursin(r"^[0-9]+\.[0-9]+\.[0-9]+$", env[key]), "invalid version metadata")
    end
    for (keys, pattern) in [(ENV_COMMITS, r"^[0-9a-f]{40}$"), (ENV_HASHES, r"^[0-9a-f]{64}$")], key in keys
        require(env[key] isa String && occursin(pattern, env[key]), "invalid digest metadata")
    end
    for key in ENV_THREADS
        require(env[key] isa Integer && !(env[key] isa Bool) && 1 <= env[key] <= 1024, "invalid thread metadata")
    end
end

function read_document(path)
    require(isfile(path) && filesize(path) <= MAX_DOCUMENT_BYTES, "missing or oversized document")
    bytes = read(path)
    value = JSON3.read(String(copy(bytes)), Dict{String,Any})
    return value, digest(bytes)
end

"""Never silently overwrite an artifact. Publish only a complete JSON document."""
function write_document(path, value)
    require(!ispath(path), "output already exists")
    mkpath(dirname(abspath(path)))
    temp, io = mktemp(dirname(abspath(path)))
    try
        JSON3.write(io, value)
        write(io, '\n')
        close(io)
        chmod(temp, 0o644)
        mv(temp, path; force=false)
    finally
        isopen(io) && close(io)
        ispath(temp) && rm(temp)
    end
end

hexfloat(x::Float64) = string(reinterpret(UInt64, x); base=16, pad=16)
function unhexfloat(value)
    require(value isa String && occursin(r"^[0-9a-f]{16}$", value), "expected binary64 hex bits")
    x = reinterpret(Float64, parse(UInt64, value; base=16))
    require(isfinite(x), "nonfinite candidate number")
    return x
end

function rational(value)
    require(value isa String && occursin(r"^-?[0-9]{1,40}/[1-9][0-9]{0,39}$", value),
        "expected bounded exact rational n/d")
    n, d = split(value, '/')
    return parse(BigInt, n) // parse(BigInt, d)
end

"""These two frozen problems are the whole input language; no eval or expressions."""
function problem_definition(id)
    common = Dict{String,Any}("schema_version" => "cap.problem/0.1", "id" => id,
        "target_epsilon" => "1/10000000000")
    if id == "scalar-cuberoot"
        merge!(common, Dict("equation" => "u^3 - 2 = 0", "domain" => "real",
            "parameters" => Dict("constant" => "2/1"),
            "representation" => Dict("basis" => "scalar", "order" => 0, "norm" => "absolute"),
            "theorem" => "radii-first-order-scalar/1", "branch" => "positive-root"))
    elseif id == "cahn-hilliard"
        merge!(common, Dict("equation" => "u'' + beta*(u-u^3-c) = 0",
            "domain" => "[0,1]; u'(0)=u'(1)=0; even 2-periodic extension",
            "parameters" => Dict("c" => "0/1", "beta" => "1285/32", "frequency" => "pi"),
            "representation" => Dict("basis" => "even-fourier", "order" => 32,
                "norm" => "abs(a0)+2*sum(abs(ak),k>=1)"),
            "theorem" => "radii-second-order-cahn-hilliard/1", "branch" => "positive-first-cosine-mode"))
    else
        throw(ArgumentError("unsupported reference problem"))
    end
    return common
end

# Equal values of equal JSON types: `32.0` is not the frozen order `32`.
same(a::AbstractDict, b::AbstractDict) = Set(keys(a)) == Set(keys(b)) && all(k -> same(a[k], b[k]), keys(a))
same(a, b) = typeof(a) == typeof(b) && a == b

function load_problem(path)
    problem, sha = read_document(path)
    require(haskey(problem, "id") && problem["id"] isa String && same(problem, problem_definition(problem["id"])),
        "problem differs from the frozen theorem specification")
    return problem, sha
end

"""Select the generic interval matrix product. Call at top level, before verifying:
the setting replaces a method, which code already running does not see."""
configure_arithmetic() = (IA.configure(matmul=:slow); nothing)

function check_arithmetic()
    require(IA.default_numtype() === Float64 && IA.default_flavor() === IA.Flavor{:set_based}() &&
        IA.default_rounding() === IA.IntervalRounding{:correct}() && IA.default_power() === IA.PowerMode{:fast}() &&
        IA.default_matmul() === IA.MatMulMode{:slow}() && RadiiPolynomial.CONV_ALGORITHM[] === :loop,
        "interval arithmetic must be configured as: " * ARITHMETIC * "; call configure_arithmetic() first")
end

function environment()
    require(VERSION == v"1.10.10", "the verifier requires Julia 1.10.10")
    require(Base.pkgversion(RadiiPolynomial) == v"0.11.10" &&
        Base.pkgversion(IntervalArithmetic) == v"1.0.12", "unexpected validation library version")
    require(Threads.nthreads() == 1, "the verifier requires JULIA_NUM_THREADS=1")
    check_arithmetic()
    BLAS.set_num_threads(1)
    lock = TOML.parsefile(joinpath(ROOT, "Manifest.toml"))
    require(lock["deps"]["RadiiPolynomial"][1]["repo-rev"] == RP_COMMIT &&
        lock["deps"]["IntervalArithmetic"][1]["repo-rev"] == IA_COMMIT,
        "validation dependency revision changed")
    # The loaded code must be the source tree this manifest pins, not merely a
    # package reporting the same version.
    for (mod, name) in ((RadiiPolynomial, "RadiiPolynomial"), (IntervalArithmetic, "IntervalArithmetic"))
        entry = lock["deps"][name][1]
        slug = Base.version_slug(Base.UUID(entry["uuid"]), Base.SHA1(entry["git-tree-sha1"]))
        require(occursin(joinpath("packages", name, slug, "src"), something(pathof(mod), "")),
            "loaded $name is not the source tree pinned by the manifest")
    end
    return Dict("julia" => string(VERSION), "radii_polynomial" => string(Base.pkgversion(RadiiPolynomial)),
        "interval_arithmetic" => string(Base.pkgversion(IntervalArithmetic)),
        "radii_commit" => RP_COMMIT, "interval_commit" => IA_COMMIT,
        "julia_threads" => Threads.nthreads(), "blas_threads" => BLAS.get_num_threads(),
        "arithmetic" => ARITHMETIC,
        "project_sha256" => file_digest(joinpath(ROOT, "Project.toml")),
        "manifest_sha256" => file_digest(joinpath(ROOT, "Manifest.toml")),
        "verifier_sha256" => file_digest(@__FILE__),
        "cli_sha256" => file_digest(joinpath(ROOT, "bin", "cap.jl")))
end

scalar_f(u) = u^3 - exact(2)
scalar_df(u) = exact(3) * u^2
ch_f(u, c, beta) = Laplacian() * u + beta * (u - u^3 - c)
ch_df(u, c, beta) = Laplacian() + Multiplication(beta * (exact(1) - exact(3) * u^2))

"""Numerical approximation only. Its flag is never used as evidence by verify_files."""
function candidate_document(problem_path)
    started = time_ns()
    problem, problem_sha = load_problem(problem_path)
    env = environment()
    if problem["id"] == "scalar-cuberoot"
        u, success = newton(u -> (scalar_f(u), scalar_df(u)), 1.0)
        coeffs = [u]
        inverse = reshape([inv(scalar_df(u))], 1, 1)
    else
        k = problem["representation"]["order"]
        c, beta = 0.0, 40.15625 # exact dyadic reference constants, approximation phase only
        initial = Sequence(evensym(Fourier(k, π)), [c; 0.45; zeros(k-1)])
        u, success = newton(u -> (ch_f(u, c, beta), ch_df(u, c, beta)), initial)
        projection = Projection(space(u))
        inverse = coefficients(inv(projection * ch_df(u, c, beta) * projection))
        coeffs = coefficients(u)
    end
    require(success && all(isfinite, coeffs) && all(isfinite, inverse), "reference Newton solve failed")
    return Dict("schema_version" => "cap.candidate/0.1", "problem_sha256" => problem_sha,
        "problem_id" => problem["id"], "encoding" => ENCODING,
        "coefficients" => hexfloat.(Float64.(coeffs)),
        "inverse_rows" => [hexfloat.(Float64.(inverse[i, :])) for i in axes(inverse, 1)],
        "approximation" => "reference-newton/1", "newton_converged" => success,
        "environment" => env, "elapsed_seconds" => (time_ns()-started)/1e9)
end

function load_candidate(path, problem, problem_sha)
    doc, sha = read_document(path)
    fields(doc, ["schema_version", "problem_sha256", "problem_id", "encoding", "coefficients",
        "inverse_rows", "approximation", "newton_converged", "environment", "elapsed_seconds"])
    metadata(doc)
    require(doc["approximation"] == "reference-newton/1" && doc["newton_converged"] isa Bool,
        "invalid approximation metadata")
    require(doc["schema_version"] == "cap.candidate/0.1" && doc["encoding"] == ENCODING &&
        doc["problem_id"] == problem["id"] && doc["problem_sha256"] == problem_sha,
        "candidate/problem identity or encoding mismatch")
    n = problem["id"] == "scalar-cuberoot" ? 1 : 33
    require(doc["coefficients"] isa Vector && length(doc["coefficients"]) == n &&
        doc["inverse_rows"] isa Vector && length(doc["inverse_rows"]) == n &&
        all(row -> row isa Vector && length(row) == n, doc["inverse_rows"]), "candidate dimensions mismatch")
    coeffs = unhexfloat.(doc["coefficients"])
    inverse = [unhexfloat(doc["inverse_rows"][i][j]) for i in 1:n, j in 1:n]
    return coeffs, inverse, sha
end

struct PseudoInverseLaplacian <: RadiiPolynomial.AbstractDiagonalOperator end
RadiiPolynomial.getcoefficient(::PseudoInverseLaplacian,
    (codom, i)::Tuple{SymmetricSpace{<:Fourier},Integer},
    (dom, j)::Tuple{SymmetricSpace{<:Fourier},Integer}) =
    (i == j) & !(i == j == 0) ? inv(-(frequency(dom) * exact(i))^2) : zero(frequency(dom))

function interval_record(x)
    x === nothing && return nothing
    valid = isbounded(x) && !isempty_interval(x) && !isnai(x)
    require(inf(x) isa Float64 && sup(x) isa Float64,
        "interval artifacts require binary64 endpoints; never round a bound during serialization")
    return Dict("encoding" => ENCODING, "lo" => valid ? hexfloat(Float64(inf(x))) : nothing,
        "hi" => valid ? hexfloat(Float64(sup(x))) : nothing, "bounded_nonempty" => valid,
        "guaranteed" => isguaranteed(x), "decoration" => string(decoration(x)))
end

admissible(x) = isguaranteed(x) && !isnai(x) && !isempty_interval(x) && isbounded(x) &&
    decoration(x) >= dac && inf(x) >= 0

"""Conservative, independently evaluated inequalities at an explicit binary64 radius.

The returned library Boolean and interval are both checked. We do not accept a
closed upper endpoint as evidence for a strict contraction. First-order Z1 is
a whole-ball bound; second-order Z1 is the derivative defect at the centre.
"""
function assess(Y, Z1, Z2, R, epsilon; second_order)
    bounds = second_order ? [Y, Z1, Z2] : [Y, Z1]
    checks = Dict("guaranteed_bounds" => all(isguaranteed, bounds),
        "admissible_bounds" => all(admissible, bounds),
        "positive_radius" => isfinite(R) && R > 0,
        "library_success" => false, "guaranteed_radius_interval" => false,
        "self_mapping" => false, "strict_contraction" => false, "physical_accuracy" => false)
    ie = nothing
    P = nothing
    contraction = nothing
    if checks["admissible_bounds"] && checks["positive_radius"]
        ie, success = second_order ? interval_of_existence(Y, Z1, Z2, R) : interval_of_existence(Y, Z1, R)
        checks["library_success"] = success
        checks["guaranteed_radius_interval"] = admissible(ie)
        r = interval(R)
        P = Y + (Z1 - exact(1))*r
        contraction = Z1
        if second_order
            P += Z2 * r^2 / exact(2)
            contraction += Z2*r
        end
        checks["self_mapping"] = isguaranteed(P) && isbounded(P) && !isempty_interval(P) && sup(P) <= 0
        checks["strict_contraction"] = admissible(contraction) && sup(contraction) < 1
        checks["physical_accuracy"] = sup(r - interval(epsilon)) <= 0
    end
    return checks, ie, P, contraction
end

function verify_files(problem_path, candidate_path)
    started = time_ns()
    problem, problem_sha = load_problem(problem_path)
    env = environment()
    coeffs, inverse, candidate_sha = load_candidate(candidate_path, problem, problem_sha)
    scalar = problem["id"] == "scalar-cuberoot"
    extra = Dict{String,Any}()
    if scalar
        u = interval(coeffs[1]); A = interval(inverse[1, 1])
        Y = abs(A*scalar_f(u))
        R = 2sup(Y) # the chosen radius is an exact binary64 point
        ball = interval(coeffs[1], R; format=:midpoint)
        Z1 = abs(interval(1) - A*scalar_df(ball))
        Z2 = interval(0)
        injective = !in_interval(0, A)
        branch = isfinite(R) && inf(u - interval(R)) > 0
    else
        k = problem["representation"]["order"]
        # Exact mathematical pi, not the binary64 approximation to pi.
        s = evensym(Fourier(k, interval(π)))
        u = Sequence(s, interval.(coeffs))
        finite_inverse = LinearOperator(s, s, interval.(inverse))
        # The projection's coefficient type must be interval too. An interval
        # frequency alone still leaves Float64 projection entries and loses NG.
        projection = Projection(s, Interval{Float64})
        c = interval(Float64, rational(problem["parameters"]["c"]))
        beta = interval(Float64, rational(problem["parameters"]["beta"]))
        A = finite_inverse + PseudoInverseLaplacian()*(interval(I)-projection)
        Y = norm(A*ch_f(u, c, beta), Ell1())
        projection3 = Projection(evensym(Fourier(3k+1, interval(π))), Interval{Float64})
        Zfinite = opnorm(projection3 - A*(ch_df(u, c, beta)*projection3), Ell1())
        Ztail = beta*(exact(1) + exact(3)*norm(u, Ell1())^2)/(interval(π)*exact(k+2))^2
        Z1 = max(Zfinite, Ztail)
        R = 10sup(Y)
        normA = max(opnorm(finite_inverse, Ell1()), interval(1)/interval(π)^2)
        Z2 = exact(3)*beta*normA*(exact(2)*norm(u, Ell1()) + interval(R))
        # A finite square product within norm < 1 of I has an injective left factor.
        injective = admissible(Zfinite) && sup(Zfinite) < 1
        branch = isfinite(R) && inf(u[1] - interval(R)/exact(2)) > 0
        extra["Z1_finite"] = interval_record(Zfinite)
        extra["Z1_tail"] = interval_record(Ztail)
        extra["norm_A"] = interval_record(normA)
    end
    checks, ie, P, contraction = assess(Y, Z1, Z2, R, rational(problem["target_epsilon"]); second_order=!scalar)
    checks["injective_inverse"] = injective
    checks["branch_condition"] = branch
    if !scalar
        checks["analytic_tail"] = extra["Z1_tail"]["guaranteed"] && extra["Z1_tail"]["bounded_nonempty"]
    end
    bounds = Dict("Y" => interval_record(Y), "Z1" => interval_record(Z1), "Z2" => interval_record(Z2))
    merge!(bounds, extra)
    proof = Dict("theorem" => problem["theorem"], "scope" => scalar ? "real_equation" : "full_neumann_steady_state",
        "norm" => problem["representation"]["norm"], "bounds" => bounds,
        "radius_cap" => hexfloat(R), "existence_radius" => hexfloat(R), "uniqueness_radius" => hexfloat(R),
        "polynomial_at_radius" => interval_record(P), "contraction_at_radius" => interval_record(contraction),
        "library_radius_interval" => interval_record(ie), "checks" => checks,
        "tail" => scalar ? "not_applicable" : "analytic_unresolved_fourier_modes")
    verified = all(values(checks))
    return Dict("schema_version" => "cap.verification/0.1", "problem_sha256" => problem_sha,
        "candidate_sha256" => candidate_sha, "problem_id" => problem["id"],
        "outcome" => verified ? "verified" : "inconclusive",
        "reasons" => sort([name for (name, ok) in checks if !ok]), "proof" => proof,
        "environment" => env, "elapsed_seconds" => (time_ns()-started)/1e9)
end

"""Fresh rigorous computation. Never trust the saved Boolean, bounds or Newton flag.

Replay requires the same recorded verifier environment and exact bound records.
Cross-platform semantic replay with different enclosures is deliberately deferred.
"""
function replay_files(problem_path, candidate_path, certificate_path)
    started = time_ns()
    saved, certificate_sha = read_document(certificate_path)
    fields(saved, ["schema_version", "problem_sha256", "candidate_sha256", "problem_id", "outcome",
        "reasons", "proof", "environment", "elapsed_seconds"])
    metadata(saved)
    require(saved["schema_version"] == "cap.verification/0.1", "expected verification record")
    fresh = verify_files(problem_path, candidate_path)
    checks = Dict("problem_identity" => saved["problem_sha256"] == fresh["problem_sha256"] && saved["problem_id"] == fresh["problem_id"],
        "candidate_identity" => saved["candidate_sha256"] == fresh["candidate_sha256"],
        "environment_identity" => saved["environment"] == fresh["environment"],
        "original_verified" => saved["outcome"] == "verified" && saved["reasons"] == [],
        "fresh_verified" => fresh["outcome"] == "verified",
        "recomputed_proof_matches" => saved["proof"] == fresh["proof"])
    return Dict("schema_version" => "cap.replay/0.1", "problem_sha256" => fresh["problem_sha256"],
        "candidate_sha256" => fresh["candidate_sha256"], "certificate_sha256" => certificate_sha,
        "problem_id" => fresh["problem_id"], "outcome" => all(values(checks)) ? "verified" : "rejected",
        "checks" => checks, "recomputed" => fresh, "elapsed_seconds" => (time_ns()-started)/1e9)
end
end
