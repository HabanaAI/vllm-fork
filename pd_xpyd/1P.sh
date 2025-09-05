mkdir -p pd_test_log

log_file="pd_test_log/prefill.log"
echo "-------------------------------------------------------------------"
echo "Being brought to background"
echo "Log will be redirect to $log_file"
echo "..."
echo "-------------------------------------------------------------------"
bash ./1p_start_prefill.sh G3D-sys03 >> $log_file 2>&1 &
