using Test
using CAPReference
using RadiiPolynomial
using IntervalArithmetic
using LinearAlgebra

const C = CAPReference
const PROBLEMS = normpath(joinpath(@__DIR__, "..", "..", "problems"))

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
            if id == "cahn-hilliard"
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
