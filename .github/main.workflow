workflow "New workflow" {
  on = "push"
  resolves = ["Check CHANGELOG"]
}

action "Check CHANGELOG" {
  uses = "actions/bin/sh@master"
  runs = "git diff --exit-code CHANGELOG"
}
