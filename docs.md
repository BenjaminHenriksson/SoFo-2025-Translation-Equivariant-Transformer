# Onion.jl Documentation

> *It has layers.*

---

## 1. Introduction

**Onion.jl** is a minimalist Julia framework for authoring and composing neural‑network **layers**—from basic linear transforms to exotic, research‑grade building‑blocks—without committing to any particular deep‑learning engine.  Every *layer* is implemented as a plain Julia `struct` whose fields are ordinary arrays or scalars, making models easy to inspect, serialise and debug.  The name reflects the central design principle: models are built as *concentric* stacks of self‑contained layers—like an onion.

Key characteristics

* **Type‑inferable layers** – all parameters live in the type domain, unlocking zero‑overhead dispatch.
* **Backend‑agnostic** – use with Flux, Lux, Knet or your own autodiff; Onion just defines the layers.
* **Batch‑shape polymorphism** – every layer works with any number of *batch* dimensions (including none).
* **File‑per‑layer** – code stays tidy; each file contains exactly one public layer definition and its helpers.
* **Scholarly hygiene** – novel or borrowed ideas must be cited inline, with links to the original paper or code.

---

## 2. Installation

```julia
using Pkg
pkg"registry add https://github.com/MurrellGroup/MurrellGroupRegistry"
pkg"add Onion"
```

Once installed you can load the package in the usual way:

```julia
julia> using Onion
```

---

## 3. Quick Start

```julia
using Onion, Random

# 3‑layer multilayer perceptron (MLP)
model = Chain(
    Dense(10 ⇒ 64, relu),
    Dense(64 ⇒ 32, relu),
    Dense(32 ⇒ 1)
)

x = rand(Float32, 10, 128)         # 128‑sample batch with 10 features each
ŷ = model(x)                       # forward pass – works with any batch dims!
```

If you prefer another syntax, any callable collection of layers (e.g. `∘` composition, `|>` piping, or Lux’s functional style) will do—the underlying objects are ordinary functions.

> **Tip** – Because every layer is a concrete type, Julia can aggressively specialise the whole model; check `@code_warntype model(x)` to see the absence of red **Any**.

---

## 4. Design Philosophy & Rules

1. **Structs only** – no functors, dicts or anonymous closures.
2. **Fully‑inferable types** – use type parameters to store things the compiler must know (e.g. activation function, input/output size).  If a field varies at run‑time, keep it out of the type.
3. **Batch‑agnostic** – rely on the provided `glut` helper to merge or split batch dims instead of hard‑coding `size` checks.
4. **One layer per file** – keeps blame clean and encourages small, composable units.
5. **Cite your sources** – a comment such as `# He et al. (2015) – Delving Deep into Rectifiers` right above the implementation is enough.

Violations will trigger a linter error in CI.

---

## 5. Library Tour

Below is a non‑exhaustive map of the current source tree.  Use it as a jumping‑off point when exploring the code base.

| Path                     | Layer / Utility             | Purpose                                                                      |
| ------------------------ | --------------------------- | ---------------------------------------------------------------------------- |
| `src/Onion.jl`           | *Entry point*               | Defines exports, includes submodules and re‑exports third‑party dependencies |
| `src/Core/parameters.jl` | `@params`, `freeze`, `thaw` | Parameter traversal utilities                                                |
| `src/Core/glut.jl`       | `glut`                      | Handles arbitrary batch‑dimension folding                                    |
| `src/Dense/dense.jl`     | `Dense`                     | Classic fully‑connected layer with optional activation                       |
| `src/Conv/conv.jl`       | `Conv`                      | ND convolution with stride and dilation support                              |
| `src/Attention/ipa.jl`   | `InvariantPointAttention`   | AlphaFold‑style SE(3) invariant attention block                              |
| …                        | …                           | …                                                                            |

> **Note** – The exact list will evolve; consult the source tree for authoritative names.

---

## 6. Extending Onion

Creating a new custom layer is straightforward:

```julia
module MyFancyLayer
using Onion, ..Utilities  # import helpers you need

struct Fancy{F,G,T}
    w :: T      # weights (e.g. Matrix)
    b :: T      # bias    (optional)
    f :: F      # forward non‑linearity (e.g. relu)
    g :: G      # gating function
end

# mandatory forward method
a (m::Fancy)(x) = m.f.(m.w * x .+ m.b) .* m.g(x)

end # module
```

1. Put the file under `src/Fancy/`.
2. Add `include("Fancy/fancy.jl")` and `export Fancy` to `src/Onion.jl`.
3. Add tests under `test/fancy.jl` covering shape‑polymorphic and edge cases.
4. Document the layer in `docs/src/layers/fancy.md`.
5. Open a pull request; CI will run style, doctest and coverage checks.

---

## 7. Documentation

Onion uses **Documenter.jl**.  After cloning the repo, you can build the docs locally with

```julia
julia --project=docs docs/make.jl
```

The hosted versions are available at

* Stable: [https://MurrellGroup.github.io/Onion.jl/stable/](https://MurrellGroup.github.io/Onion.jl/stable/)
* Dev:    [https://MurrellGroup.github.io/Onion.jl/dev/](https://MurrellGroup.github.io/Onion.jl/dev/)

If you add a new layer, create a corresponding markdown page under `docs/src/layers/` and link it from `docs/src/index.md`.

---

## 8. Testing

`Pkg.test("Onion")` runs an exhaustive suite covering type stability, gradient correctness (via Zygote & ReverseDiff), and GPU equivalence when CUDA is available.

* `test/layers/` – shape & numerical checks per layer.
* `test/integration/` – behaviour of common layer stacks.

CI runs on Linux, macOS and Windows across Julia 1.10‑LTS and nightly.

---

## 9. Benchmarks

Benchmarks are conducted with **BenchmarkTools.jl** and aggregated by **PkgBenchmark**.  To reproduce:

```julia
julia> using PkgBenchmark
julia> benchmarkpkg("Onion"; verbose = true)
```

Results are uploaded to the GitHub Pages site for each commit.

---

## 10. Contributing

We welcome PRs for new layers, performance improvements, bug fixes and documentation enhancements.  Before opening a PR please:

1. File an issue describing the proposed change.
2. Ensure your code passes `\] test` and `\] format`, and includes unit tests.
3. Adhere to the **Rules** listed above.
4. Add yourself to `CONTRIBUTORS.md`.

---

## 11. License

Onion.jl is released under the MIT License.  See `LICENSE` for details.

---

*Happy hacking, and peel away!*
