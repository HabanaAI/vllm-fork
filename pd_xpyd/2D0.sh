mkdir -p pd_test_log

log_file="pd_test_log/decode_2d0.log"
echo "-------------------------------------------------------------------"
echo "Being brought to background"
echo "Log will be redirect to $log_file"
echo "..."
echo "-------------------------------------------------------------------"
MAX_SIZE=$((50*1024*1024))
if [ -f "$log_file" ]; then
  cur_size=$(stat -c%s "$log_file" 2>/dev/null || echo 0)
  if [ "$cur_size" -gt "$MAX_SIZE" ]; then
    ts=$(TZ="Asia/Shanghai" date +"%Y%m%d_%H%M%S")
    mv "$log_file" "${log_file}.${ts}"
    : > "$log_file"
  fi
fi
## Usage: ./2D0.sh [HOSTNAME]
## Forward optional hostname to dp0_xp2d_start_decode.sh (as its second arg)
# If a hostname is provided, pass it through; otherwise call without it
if [ -n "$1" ]; then
  bash dp0_xp2d_start_decode.sh 1 "$1" >> $log_file 2>&1 &
else
  bash dp0_xp2d_start_decode.sh 1 >> $log_file 2>&1 &
fi
