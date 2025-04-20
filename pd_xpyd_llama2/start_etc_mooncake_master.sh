pkill etcd
pkill mooncake_master
sleep 5
etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://localhost:2379  >etcd.log 2>&1 &
mooncake_master -enable_gc true -port 50001 &> mooncake_master.log & # -v 2

