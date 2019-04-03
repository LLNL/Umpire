workflow "New workflow" {
  on = "push"
  resolves = ["Check CHANGELOG"]
}

action "Check CHANGELOG" {
  uses = "./.github/actions/bin/diff-check"
  args = ["CHANGELOG"]
}
