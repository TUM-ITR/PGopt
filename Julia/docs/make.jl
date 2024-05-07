using Documenter, PGopt

makedocs(
    sitename = " ",
    pages = [
        "index.md",
        "list_of_functions.md",
        "reference.md"
    ]
)

deploydocs(
    repo = "github.com/TUM-ITR/PGopt.git",
)