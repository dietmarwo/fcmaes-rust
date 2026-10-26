#!/usr/bin/env julia
using CAPReference

function main(args)
    # A startup file can change the interval arithmetic or anything else the
    # verifier relies on before it runs.
    Base.JLOptions().startupfile == 2 || throw(ArgumentError("run cap.jl with julia --startup-file=no"))
    isempty(args) && throw(ArgumentError("usage: cap.jl candidate PROBLEM OUT | verify PROBLEM CANDIDATE OUT | replay PROBLEM CANDIDATE CERTIFICATE OUT"))
    command = first(args)
    if command == "candidate" && length(args) == 3
        result = CAPReference.candidate_document(args[2])
    elseif command == "verify" && length(args) == 4
        result = CAPReference.verify_files(args[2], args[3])
    elseif command == "replay" && length(args) == 5
        result = CAPReference.replay_files(args[2], args[3], args[4])
    else
        throw(ArgumentError("unknown command or wrong argument count"))
    end
    CAPReference.write_document(last(args), result)
    println(get(result, "outcome", "candidate_written"))
    # A valid inconclusive verification is a completed computation. Replay is
    # the strict final acceptance gate; retain its output even on rejection.
    return command == "replay" && result["outcome"] != "verified" ? 3 : 0
end

try
    # The arithmetic setting replaces a method, which code already running
    # would not see, so `main` is entered through `invokelatest`.
    CAPReference.configure_arithmetic()
    exit(Base.invokelatest(main, ARGS))
catch exception
    showerror(stderr, exception)
    println(stderr)
    exit(2)
end
