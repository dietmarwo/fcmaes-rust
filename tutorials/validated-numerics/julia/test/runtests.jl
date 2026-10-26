using Test
using CAPReference
using RadiiPolynomial
using IntervalArithmetic
using LinearAlgebra

const C = CAPReference
const PROBLEMS = normpath(joinpath(@__DIR__, "..", "..", "problems"))
C.configure_arithmetic()  # at top level, as cap.jl does; later testsets see it

@testset "Arithmetic configuration and provenance gate" begin
    @test C.environment()["arithmetic"] == C.ARITHMETIC
    # Round-to-nearest still reports guaranteed intervals; the gate must refuse it.
    IntervalArithmetic.configure(rounding=:none)
    try
        @test isguaranteed(Base.invokelatest(() -> interval(1.0) + interval(1e-30)))
        @test_throws ArgumentError Base.invokelatest(C.environment)
    finally
        IntervalArithmetic.configure(rounding=:correct)
    end
    @test Base.invokelatest(C.environment)["arithmetic"] == C.ARITHMETIC
    @test in_interval(big(π), interval(π))
end

@testset "Library behavior the proof relies on" begin
    s = evensym(Fourier(1, 1.0))
    # Ell1 weighs mode 0 by 1 and modes >= 1 by 2, in operator norms too.
    @test opnorm(LinearOperator(s, s, [0.0 0.0; 1.0 0.0]), Ell1()) == 2
    @test opnorm(LinearOperator(s, s, [0.0 1.0; 0.0 0.0]), Ell1()) == 0.5
    # A finite operator truncates a longer input to its domain: A_finite ∘ Π_K.
    b = Sequence(evensym(Fourier(3, 1.0)), [1.0, 2.0, 3.0, 4.0])
    @test coefficients(LinearOperator(s, s, [1.0 0.0; 0.0 1.0]) * b) == [1.0, 2.0]
    # The finite defect keeps every output row of DF on input modes up to 3K+1.
    K = 2
    sK = evensym(Fourier(K, interval(π)))
    u = Sequence(sK, interval.([0.1, 0.2, 0.3]))
    finite = LinearOperator(sK, sK, interval.(Matrix{Float64}(I, K+1, K+1)))
    A = finite + C.PseudoInverseLaplacian()*(interval(I) - Projection(sK, Interval{Float64}))
    projection3 = Projection(evensym(Fourier(3K+1, interval(π))), Interval{Float64})
    defect = projection3 - A*(C.ch_df(u, interval(0.0), interval(1.0))*projection3)
    @test size(coefficients(defect)) == (5K+2, 3K+2)
end

@testset "Lossless data and exact constants" begin
    for x in [0.0, -0.0, nextfloat(0.0), 0.1, -3.5, floatmax(Float64)]
        @test isequal(C.unhexfloat(C.hexfloat(x)), x)
    end
    for text in ["0.1", "7ff0000000000000", "7ff8000000000000", "000000000000000G"]
        @test_throws ArgumentError C.unhexfloat(text)
    end
    @test C.rational("1285/32") == 1285//32
    @test_throws ArgumentError C.rational("1/0")
    @test_throws ArgumentError C.rational("0.1")
    exact_tenth = C.rational("1/10")
    @test in_interval(exact_tenth, interval(exact_tenth))
    @test !in_interval(exact_tenth, interval(0.1))
    @test_throws ArgumentError C.interval_record(interval(BigFloat, exact_tenth))
end

@testset "Theorem gate, not a success Boolean" begin
    # This example has a positive library Boolean while the interval is NG.
    Y = convert(Interval{Float64}, 1e-12)
    Z1 = interval(0.1)
    ie, success = interval_of_existence(Y, Z1, 1e-10)
    @test success
    @test !isguaranteed(ie)
    checks, _, _, _ = C.assess(Y, Z1, interval(0), 1e-10, 1//100; second_order=false)
    @test !all(values(checks))
    @test !checks["guaranteed_bounds"]
    for (y,z1,z2,r) in [(interval(1),interval(0.1),interval(1),0.01),
                        (interval(1e-12),interval(0.9,1.1),interval(1),1e-10),
                        (interval(1e-12),interval(0),interval(1),1.0),
                        (emptyinterval(Float64),interval(0),interval(0),1e-10),
                        (interval(0,Inf),interval(0),interval(0),1e-10),
                        (nai(Float64),interval(0),interval(0),1e-10)]
        checks, _, _, _ = C.assess(y,z1,z2,r,1//100; second_order=true)
        @test !all(values(checks))
    end
    # Z2 = 0 takes the first-order library path, while our explicit checks still apply.
    checks, _, _, _ = C.assess(interval(1e-12),interval(0.1),interval(0),1e-10,1//100; second_order=true)
    @test all(values(checks))
    checks, _, _, _ = C.assess(interval(1e-12),interval(0.1),interval(0),1e-10,1//10^12; second_order=true)
    @test !checks["physical_accuracy"]
    # Self-mapping holds at R but contraction fails there, although the library
    # finds a smaller radius: R itself must satisfy both conditions.
    checks, _, _, _ = C.assess(interval(1e-20),interval(0.5),interval(9e9),1e-10,1//100; second_order=true)
    @test checks["self_mapping"] && !checks["strict_contraction"]
end

@testset "Reference verification, replay and refutations" begin
    mktempdir() do directory
        for id in ["scalar-cuberoot", "cahn-hilliard"]
            problem = joinpath(PROBLEMS, id*".json")
            candidate = joinpath(directory, id*"-candidate.json")
            certificate = joinpath(directory, id*"-verification.json")
            doc = C.candidate_document(problem)
            C.write_document(candidate, doc)
            verified = C.verify_files(problem, candidate)
            @test verified["outcome"] == "verified"
            @test all(values(verified["proof"]["checks"]))
            @test C.unhexfloat(verified["proof"]["existence_radius"]) <= 1e-10
            C.write_document(certificate, verified)
            replay = C.replay_files(problem,candidate,certificate)
            @test replay["outcome"] == "verified"
            @test replay["recomputed"]["proof"] == verified["proof"]
            @test_throws ArgumentError C.write_document(certificate, verified)

            for mutation in [v -> (v["proof"]["checks"]["strict_contraction"] = false),
                             v -> (v["proof"]["bounds"]["Y"]["hi"] = C.hexfloat(0.0)),
                             v -> (v["environment"]["manifest_sha256"] = "0"^64),
                             v -> (v["candidate_sha256"] = "0"^64)]
                tampered = deepcopy(verified); mutation(tampered)
                path = tempname(directory); C.write_document(path,tampered)
                @test C.replay_files(problem,candidate,path)["outcome"] == "rejected"
            end

            modified = deepcopy(doc)
            modified["coefficients"][1] = C.hexfloat(0.25)
            path = tempname(directory); C.write_document(path,modified)
            @test C.replay_files(problem,path,certificate)["outcome"] == "rejected"

            modified = deepcopy(doc); modified["newton_converged"] = false
            path = tempname(directory); C.write_document(path,modified)
            @test C.verify_files(problem,path)["outcome"] == "verified" # re-proved, never trust that flag

            modified = deepcopy(doc); modified["inverse_rows"] = fill(fill(C.hexfloat(0.0),length(doc["coefficients"])),length(doc["coefficients"]))
            path = tempname(directory); C.write_document(path,modified)
            @test C.verify_files(problem,path)["outcome"] == "inconclusive"

            for mutation in [d -> push!(d["coefficients"], C.hexfloat(0.0)),
                             d -> (d["coefficients"][1] = "7ff8000000000000"),
                             d -> (d["problem_sha256"] = "0"^64),
                             d -> (d["elapsed_seconds"] = -1),
                             d -> (d["unrecognized"] = true)]
                modified = deepcopy(doc); mutation(modified)
                path = tempname(directory); C.write_document(path,modified)
                @test_throws ArgumentError C.verify_files(problem,path)
            end
            wrong = C.problem_definition(id); wrong["representation"]["order"] = float(wrong["representation"]["order"])
            path = tempname(directory); C.write_document(path,wrong)
            @test_throws ArgumentError C.verify_files(path,candidate) # 32.0 is not the frozen order 32
            if id == "cahn-hilliard"
                # With c = 0, -ū is a solution too; only the branch condition tells them apart.
                negated = deepcopy(doc); negated["coefficients"] = [C.hexfloat(-C.unhexfloat(x)) for x in doc["coefficients"]]
                path = tempname(directory); C.write_document(path,negated)
                @test C.verify_files(problem,path)["reasons"] == ["branch_condition"]
                @test C.unhexfloat(verified["proof"]["bounds"]["Z1_tail"]["hi"]) > 0
                missing_tail = deepcopy(verified); delete!(missing_tail["proof"]["bounds"],"Z1_tail")
                path = tempname(directory); C.write_document(path,missing_tail)
                @test C.replay_files(problem,candidate,path)["outcome"] == "rejected"
                @test norm(Sequence(evensym(Fourier(1,π)),[0.0,1.0]),Ell1()) == 2
                wrong = C.problem_definition(id); wrong["parameters"]["frequency"] = "3.141592653589793"
                path = tempname(directory); C.write_document(path,wrong)
                @test_throws ArgumentError C.verify_files(path,candidate)
            end
        end
    end
end

include("searchtests.jl")
