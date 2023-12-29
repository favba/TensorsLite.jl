using TensorsLite
using Documenter

DocMeta.setdocmeta!(TensorsLite, :DocTestSetup, :(using TensorsLite); recursive=true)

makedocs(;
    modules=[TensorsLite],
    authors="Felipe A. V. de Bragança Alves <favbalves@gmail.com> and contributors",
    repo="https://github.com/favba/TensorsLite.jl/blob/{commit}{path}#{line}",
    sitename="TensorsLite.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://favba.github.io/TensorsLite.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/favba/TensorsLite.jl",
    devbranch="main",
)
