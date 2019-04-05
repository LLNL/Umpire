workflow "New workflow" {
  on = "push"
  resolves = ["Check CHANGELOG", "Static Analysis", "Clang Tidy"]
}

action "Check CHANGELOG" {
  uses = "./.github/actions/bin/diff-check"
  args = ["CHANGELOG"]
}

action "Static Analysis" {
  uses = "./.github/actions/static-analysis"
}

action "Clang Tidy" {
  uses = "./.github/actions/clang-tidy"
}
