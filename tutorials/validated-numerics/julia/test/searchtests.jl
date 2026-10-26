include(joinpath(@__DIR__,"../src/CAPSearch.jl"))
const S = CAPSearch

@testset "Persistent search model preserves the fixed target" begin
    for id in ["scalar-cuberoot","cahn-hilliard"]
        problem=joinpath(PROBLEMS,id*".json")
        m=S.model(problem)
        n=S.dimension(m)
        nominal=ones(n)
        before=deepcopy(m.candidate)
        first=S.estimate(m,nominal)
        S.estimate(m,fill(0.7,n))
        @test isequal(first,S.estimate(m,nominal))
        @test m.candidate==before
        @test m.inverse==[C.unhexfloat(before["inverse_rows"][i][j]) for i in axes(m.inverse,1),j in axes(m.inverse,2)]
        @test first["score"] < 0
        @test_throws ArgumentError S.estimate(m,fill(NaN,n))
        @test_throws ArgumentError S.estimate(m,fill(0.49,n))
        @test_throws ArgumentError S.estimate(m,zeros(n+1))
        candidate=S.candidate(m,fill(0.9,n))
        @test candidate["coefficients"]==before["coefficients"]
        @test candidate["problem_sha256"]==before["problem_sha256"]
        @test candidate["inverse_rows"]!=before["inverse_rows"]
        mktempdir() do directory
            path=joinpath(directory,"candidate.json")
            C.write_document(path,candidate)
            # A numerical estimate never substitutes for this rigorous check.
            @test C.verify_files(problem,path)["outcome"]=="verified"
        end
    end
end

@testset "Bounded framed server input" begin
    @test S.read_frame(IOBuffer("{\"x\":1}\n"))["x"]==1
    @test S.read_frame(IOBuffer(""))===nothing
    @test_throws ArgumentError S.read_frame(IOBuffer("{\"x\":1}"))
    @test_throws ArgumentError S.read_frame(IOBuffer("0123456789\n");limit=4)
    @test_throws Exception S.read_frame(IOBuffer("no-json\n"))
end
