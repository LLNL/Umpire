workflow "New workflow" {
  on = "pull_request"
  resolves = ["Check CHANGELOG"]
}

action "Check CHANGELOG" {
  uses = "actions/bin/sh@master"
  runs = "git diff --exit-code CHANGELOG"
}
