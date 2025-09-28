mkdir -p pd_test_log

log_file="pd_test_log/decode_2d0.log"
echo "-------------------------------------------------------------------"
echo "Being brought to background"
echo "Log will be redirect to $log_file"
echo "..."
echo "-------------------------------------------------------------------"
bash dp0_xp2d_start_decode.sh >> $log_file 2>&1 &
