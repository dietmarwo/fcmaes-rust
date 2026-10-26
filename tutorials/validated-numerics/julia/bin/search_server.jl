#!/usr/bin/env julia
using CAPReference
include(joinpath(@__DIR__,"../src/CAPSearch.jl"))

function main(args)
    Base.JLOptions().startupfile == 2 || throw(ArgumentError("use --startup-file=no"))
    length(args) == 1 || throw(ArgumentError("usage: search_server.jl PROBLEM"))
    CAPSearch.serve(only(args))
end

try
    CAPReference.configure_arithmetic()
    Base.invokelatest(main,ARGS)
catch exception
    showerror(stderr,exception)
    println(stderr)
    exit(2)
end
