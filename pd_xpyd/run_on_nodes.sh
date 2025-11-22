#!/usr/bin/env bash

set -euo pipefail

BASE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [ $# -lt 2 ]; then
  echo "Usage: $0 <env_file> <command> [args...]" >&2
  exit 1
fi

ENV_FILE=$1
shift

if [[ "$ENV_FILE" != /* ]]; then
  ENV_FILE="$BASE_DIR/$ENV_FILE"
fi

if [ ! -f "$ENV_FILE" ]; then
  echo "Environment file $ENV_FILE not found" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$ENV_FILE"

if ! declare -p ROLE_IP >/dev/null 2>&1; then
  echo "ROLE_IP associative array must be defined in $ENV_FILE" >&2
  exit 1
fi


COMMAND_ARR=("$@")
# Allow users to provide the command either as separate arguments or as a
# single quoted string. If only one argument is present, attempt to split it
# using shell-style parsing so that `run_on_nodes.sh env.sh "mv a b"` works the
# same as `run_on_nodes.sh env.sh mv a b`.
if [ ${#COMMAND_ARR[@]} -eq 1 ]; then
  if command -v python3 >/dev/null 2>&1; then
    mapfile -t COMMAND_ARR < <(
      python3 -c 'import shlex, sys; [print(tok) for tok in shlex.split(sys.argv[1])]' \
        "${COMMAND_ARR[0]}"
    )
  else
    echo "python3 is required to parse the command string when provided as a single argument" >&2
    exit 1
  fi
fi

if [ ${#COMMAND_ARR[@]} -eq 0 ]; then
  echo "No command provided" >&2
  exit 1
fi

mapfile -t ROLE_KEYS < <(printf '%s\n' "${!ROLE_IP[@]}" | sort)

if [ ${#ROLE_KEYS[@]} -eq 0 ]; then
  echo "No ROLE_IP entries found in $ENV_FILE" >&2
  exit 1
fi

for key in "${ROLE_KEYS[@]}"; do
  ip="${ROLE_IP[$key]}"
  host="${ROLE_HOST[$key]:-$ip}"

  if [ -z "$ip" ]; then
    echo "Skipping $key: IP not defined" >&2
    continue
  fi

  echo "---- Executing on $key (host=$host, ip=$ip) ----"
  ssh root@"$ip" bash -s -- "$BASE_DIR" "${COMMAND_ARR[@]}" <<'EOF'
set -euo pipefail
cd "$1"
shift
"$@"
EOF
  echo
 done
