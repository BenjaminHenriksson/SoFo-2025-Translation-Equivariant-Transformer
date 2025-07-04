using Pkg; Pkg.activate(".")

# Make sure the Dev version of Onion is loaded (should print "dev env" when loading)
using Flux, Onion, Test

using Random

# These tests should actually be EQUIVARIANT, not INVARIANT!
# Tests need to be rewritten to consider RoPE between tokens, not independently of postions!

