workflow "New workflow" {
  on = "push"
  resolves = ["Check CHANGELOG"]
}

action "Check CHANGELOG" {
  uses = "veggiemonk/bin/git@04d5cf45890e02c5659bd47b35f5a3e540e0b108"
  args = ["diff --exit-code CHANGELOG"]
}
