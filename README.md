# Onion

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://MurrellGroup.github.io/Onion.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MurrellGroup.github.io/Onion.jl/dev/)
[![Build Status](https://github.com/MurrellGroup/Onion.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MurrellGroup/Onion.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/MurrellGroup/Onion.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MurrellGroup/Onion.jl)

## Installation

```julia
using Pkg
pkg"registry add https://github.com/MurrellGroup/MurrellGroupRegistry
pkg"add Onion"
```

## Rules

- Layers should be structs.
- Layers should have fully inferrable types (use type parameters)
- Layers should work with any number (including zero) batch dims (tip: use `glut`)
- Put each layer in a separate file.
- Cite. If from a paper/code include a link/ref in a comment.
