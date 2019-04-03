workflow "New workflow" {
  on = "push"
  resolves = ["Check CHANGELOG"]
}

action "Check CHANGELOG" {
  uses = "./github/actions/bin/sh"
  args = "git diff --exit-code CHANGELOG; exit `expr $? - 1`"
}
