using Onion
using Documenter

DocMeta.setdocmeta!(Onion, :DocTestSetup, :(using Onion); recursive=true)

makedocs(;
    modules=[Onion],
    authors="murrellb <murrellb@gmail.com> and contributors",
    sitename="Onion.jl",
    format=Documenter.HTML(;
        canonical="https://MurrellGroup.github.io/Onion.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/MurrellGroup/Onion.jl",
    devbranch="main",
)
