# Maintaining host-configs with `rad-bat`

Ordinary Umpire CI sets `RAD_BAT_MODE=host-config`. Every build-and-test job
uses the stable host-config filename stored beside its `SPEC`; a missing file
is an error and never causes an implicit Spack invocation.

To regenerate the complete maintained set, start a GitLab pipeline for the
mirrored GitHub branch with:

- `RAD_BAT_MODE=spack`
- optionally `RAD_BAT_PUSH_TO_BUILDCACHE=true` to publish dependencies

Every job creates an immutable, project-specific dependency installation and
publishes its generated host-config as a job artifact. Download all artifacts
into `host-configs/`, review them, and commit them through the normal GitHub
branch and pull-request workflow. The GitLab mirror is never modified by the
refresh pipeline.
