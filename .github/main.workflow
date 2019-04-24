workflow "New workflow" {
  on = "push"
  resolves = ["Check CHANGELOG", "Static Analysis"]
}

action "Check CHANGELOG" {
  uses = "./.github/actions/bin/diff-check"
  args = ["CHANGELOG.md"]
}

action "Static Analysis" {
  uses = "./.github/actions/static-analysis"
}


workflow "on pull request merge, delete the branch" {
  on = "pull_request"
  resolves = ["Delete merged branch"]
}

action "Delete merged branch" {
  uses = "jessfraz/branch-cleanup-action@master"
  secrets = ["GITHUB_TOKEN"]
}
