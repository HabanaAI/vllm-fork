workdir=/host/mnt/ctrl/disk1/kf/vllm-fork-kf/pd_xpyd

# Define role -> hostname and IP defaults
source env.sh

# Parse options for P benchmark mode (default: off). Usage: ./1P2D.sh [-b]
P_ARGS=""
while getopts ":b" opt; do
  case $opt in
    b)
      P_ARGS="-b"
      ;;
    \?)
      echo "Unknown option: -$OPTARG" >&2
      ;;
  esac
done

# Shift out parsed options and capture optional HOSTNAME argument
shift $((OPTIND -1))
# Default to empty; 1P.sh and dp scripts will use defaults when empty
P_HOSTNAME_ARG=${1:-""}
D0_HOSTNAME_ARG=${2:-""}
D1_HOSTNAME_ARG=${3:-""}

# Resolve effective hostnames using defaults
P_HOST=${P_HOSTNAME_ARG:-${ROLE_HOST[P]}}
D0_HOST=${D0_HOSTNAME_ARG:-${ROLE_HOST[D0]}}
D1_HOST=${D1_HOSTNAME_ARG:-${ROLE_HOST[D1]}}



if [ -n "$P_ARGS" ]; then
  echo "Start 1P on current node (benchmark mode)"
  bash ./1P.sh "$P_HOST" benchmark
else
  echo "Start 1P on current node"
  bash ./1P.sh "$P_HOST"
fi
sleep 1
echo "[SSH] Start 2D0 on 10.112.242.153 node"
ssh root@${ROLE_IP[D0]} "cd $workdir; USR_DP_MASTER_IP=${ROLE_IP[D0]} bash ./2D0.sh $D0_HOST"
sleep 1
echo "[SSH] Start 2D1 on 10.112.242.24 node"
ssh root@${ROLE_IP[D1]} "cd $workdir; USR_DP_MASTER_IP=${ROLE_IP[D0]} bash ./2D1.sh $D1_HOST"
