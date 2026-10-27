#!/usr/bin/env bash
set -euo pipefail
# Isaac's interpreter is root-accessible in this image. Restore ownership only
# of root-owned outputs in our designated writable directories, also on failure.
restore_ownership() {
  local result=$?
  trap - EXIT
  local directory
  for directory in \
    .deps-go2-4090 .docker-home .pip-tmp logs wandb plt \
    offline_world/ckpt/4090 offline_world/ckpt/4090-smoke \
    offline_agent/ckpt/4090 offline_agent/ckpt/4090-smoke; do
    if [[ -d "/workspace/$directory" ]]; then
      find "/workspace/$directory" -xdev -uid 0 \
        -exec chown -h "${NEUBAY_HOST_UID:?}:${NEUBAY_HOST_GID:?}" {} + || \
        echo "Could not restore all ownership under $directory" >&2
    fi
  done
  exit "$result"
}
trap restore_ownership EXIT
"$@"
