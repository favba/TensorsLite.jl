using VectorBasis
using Documenter

DocMeta.setdocmeta!(VectorBasis, :DocTestSetup, :(using VectorBasis); recursive=true)

makedocs(;
    modules=[VectorBasis],
    authors="Felipe A. V. de Bragança Alves <favbalves@gmail.com> and contributors",
    repo="https://github.com/favba/VectorBasis.jl/blob/{commit}{path}#{line}",
    sitename="VectorBasis.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://favba.github.io/VectorBasis.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/favba/VectorBasis.jl",
    devbranch="main",
)
