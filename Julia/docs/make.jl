using Documenter, PGopt

makedocs(
    sitename = " ",
    pages = [
        "index.md",
        "Examples" => [
            "examples/autocorrelation.md",
            "examples/PG_OCP_known_basis_functions.md",
            "examples/PG_OCP_generic_basis_functions.md",
           ],
        "list_of_functions.md",
        "reference.md"
    ]
)

deploydocs(
    repo = "github.com/TUM-ITR/PGopt.git",
)