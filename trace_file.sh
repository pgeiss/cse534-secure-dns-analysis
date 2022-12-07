filename=$1;
pcap_file_name=0;

for i in $(cat data/sites/$filename.txt); do
    file_path=data/traces/$filename/$pcap_file_name.pcap;
    touch $file_path;

    # NOTE: On MacOS this generates a pcapng by default. 
    #       On linux, it generates a pcap by default.
    # Uncomment for DOQ capture sudo tcpdump -n udp -SX -i any port $2 -w $file_path & dump_pid=$!;
    sudo tcpdump -SX -i any host dns.adguard.com and port $2 -w $file_path & dump_pid=$!;
    /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome $i;
    sleep 5;
    sudo kill $dump_pid;
    sudo chown pgeiss:staff $file_path;
    ((pcap_file_name++))
done;