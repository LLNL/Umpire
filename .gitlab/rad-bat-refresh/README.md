# Refreshing maintained host-configs

Start a GitLab pipeline for the mirrored GitHub branch with:

- `RAD_BAT_CI_WORKFLOW=refresh`
- optionally `RAD_BAT_REFRESH_TARGET=<machine>/<configuration>`

With no target, all entries in this directory run. A selected target runs only
that configuration, for example `dane/gcc_11_2_1_numa`.

Each job creates a new immutable dependency installation below the Umpire
service workspace and publishes its generated host-config as a job artifact.
Download the artifact into a local GitHub checkout, review it, and commit it to
the GitHub branch. The GitLab mirror is never modified by the refresh job.
