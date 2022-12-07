import sys
import dpkt
import numpy as np
import scipy.stats

# This file should not be imported.
if __name__ != '__main__':
    print('Do not import `trace_analysis`.')
    sys.exit(1)


#### Constants ####
BASE_FILE_PATH = 'data/traces/'
CATEGORY_DCT = {
    'arts_entertainment': 0,
    'business': 1,
    'community': 2,
    'ecommerce': 3,
    'finance': 4,
    'food': 5,
    'gaming': 6,
    'news': 7,
    'sports': 8,
    'technology': 9,
    'travel': 10
}

#### Functions ####
def get_statistics(ls):
    # We need to get max/min/mean/median/stddev/variance/coeff for a few things.
    #   Numpy provides all of these values right out of the box. Handy.

    # Additionally, we need entropy. Scipy handles this.
    arr = np.array(ls)

    max_val = np.max(arr)
    min_val = np.min(arr)
    mean = np.mean(arr)
    median = np.median(arr)
    stddev = np.std(arr)
    variance = np.var(arr)
    coeff = stddev / mean
    entropy = scipy.stats.entropy(ls)
    return (max_val, min_val, mean, median, stddev, variance, coeff, entropy)

def process_pcap(pcap_file):
    # The pcapng files contain Ethernet frames, the inner frames don't have much
    #   useful information since they're encrpyted.
    # eth.data == IP
    # eth.data.data == UDP if DOQ, TCP if DOH
    pcap = dpkt.pcapng.Reader(pcap_file)
    time_pkts = [(timestamp, dpkt.ethernet.Ethernet(buf)) for timestamp, buf in pcap]
    times = [time for (time, _) in time_pkts]
    pkts = [pkt for (_, pkt) in time_pkts]
    is_udp = isinstance(pkts[0].data.data, dpkt.udp.UDP)

    # We need to be consistent with in/out packets.
    # eth.data.src and eth.data.dst are always consistent so we'll use that.
    src = pkts[0].data.src
    dst = pkts[0].data.dst

    out_time_pkts = [time_pkt for time_pkt in time_pkts if time_pkt[1].data.src == src]
    out_pkts = [pkt for (_, pkt) in out_time_pkts]
    out_times = [time for (time, _) in out_time_pkts]

    in_time_pkts = [time_pkt for time_pkt in time_pkts if time_pkt[1].data.src == dst]
    in_pkts = [pkt for (_, pkt) in in_time_pkts]
    in_times = [time for (time, _) in in_time_pkts]

    # In/out Packets
    out_pkt_cnt = len(out_pkts)
    in_pkt_cnt = len(in_pkts)

    # Packet Length (max, min, mean, median, stddev, variance, coeff variation)
    # Note: coeff variation is stddev / mean
    if is_udp:
        out_pkt_lengths = [pkt.data.data.ulen for pkt in out_pkts]
        in_pkt_lengths = [pkt.data.data.ulen for pkt in in_pkts]
    else:
        out_pkt_lengths = [len(pkt.data.data) for pkt in out_pkts]
        in_pkt_lengths = [len(pkt.data.data) for pkt in in_pkts]
    
    (
        out_pkt_length_max,
        out_pkt_length_min,
        out_pkt_length_mean,
        out_pkt_length_median,
        out_pkt_length_stddev,
        out_pkt_length_variance,
        out_pkt_length_coeff,
        out_pkt_length_entropy
    ) = get_statistics(out_pkt_lengths)

    (
        in_pkt_length_max,
        in_pkt_length_min,
        in_pkt_length_mean,
        in_pkt_length_median,
        in_pkt_length_stddev,
        in_pkt_length_variance,
        in_pkt_length_coeff,
        in_pkt_length_entropy
    ) = get_statistics(in_pkt_lengths)

    # Inter-arrival Time (same stats)
    out_interarrival_times = np.diff(np.array(out_times))
    in_interarrival_times = np.diff(np.array(in_times))

    (
        out_interarrival_time_max,
        out_interarrival_time_min,
        out_interarrival_time_mean,
        out_interarrival_time_median,
        out_interarrival_time_stddev,
        out_interarrival_time_variance,
        out_interarrival_time_coeff,
        out_interarrival_time_entropy
    ) = get_statistics(out_interarrival_times)

    (
        in_interarrival_time_max,
        in_interarrival_time_min,
        in_interarrival_time_mean,
        in_interarrival_time_median,
        in_interarrival_time_stddev,
        in_interarrival_time_variance,
        in_interarrival_time_coeff,
        in_interarrival_time_entropy
    ) = get_statistics(in_interarrival_times)

    # Duration
    out_pkt_duration = out_times[-1] - out_times[0]
    in_pkt_duration = in_times[-1] - in_times[0]

    # Cumulative Bytes
    out_pkt_bytes = sum(out_pkt_lengths)
    in_pkt_bytes = sum(in_pkt_lengths)

    # Rate of Bytes Sent
    out_pkt_rate = out_pkt_bytes / out_pkt_duration
    in_pkt_rate = in_pkt_bytes / in_pkt_duration

    # Throughput
    out_pkt_throughput = out_pkt_cnt / out_pkt_duration
    in_pkt_throughput = in_pkt_cnt / in_pkt_duration

    return [
        out_pkt_cnt,
        in_pkt_cnt,
        out_pkt_length_max,
        out_pkt_length_min,
        out_pkt_length_mean,
        out_pkt_length_median,
        out_pkt_length_stddev,
        out_pkt_length_variance,
        out_pkt_length_coeff,
        in_pkt_length_max,
        in_pkt_length_min,
        in_pkt_length_mean,
        in_pkt_length_median,
        in_pkt_length_stddev,
        in_pkt_length_variance,
        in_pkt_length_coeff,
        out_interarrival_time_max,
        out_interarrival_time_min,
        out_interarrival_time_mean,
        out_interarrival_time_median,
        out_interarrival_time_stddev,
        out_interarrival_time_variance,
        out_interarrival_time_coeff,
        in_interarrival_time_max,
        in_interarrival_time_min,
        in_interarrival_time_mean,
        in_interarrival_time_median,
        in_interarrival_time_stddev,
        in_interarrival_time_variance,
        in_interarrival_time_coeff,
        out_pkt_duration,
        in_pkt_duration,
        out_pkt_bytes,
        in_pkt_bytes,
        out_pkt_rate,
        in_pkt_rate,
        out_pkt_length_entropy,
        out_interarrival_time_entropy,
        in_pkt_length_entropy,
        in_interarrival_time_entropy,
        out_pkt_throughput,
        in_pkt_throughput
    ]

#### Main ####
if len(sys.argv) != 3:
    print('Usage: python3 trace_analysis.py <dirname> <doq|doh>')
    sys.exit(0)

category_name = sys.argv[1]
doq_doh = sys.argv[2]
dirname = f'{category_name}/{doq_doh}'
file_path = BASE_FILE_PATH + dirname
category = CATEGORY_DCT[category_name]

csv_rows = []
for i in range(0, 50):
    try:
        file = open(f'{file_path}/{i}.pcap', 'rb')
    except (IOError, FileNotFoundError):
        # File doesn't exist, just continue the loop
        continue
    row = process_pcap(file)
    row.append(category)
    csv_rows.append(row)

outfile = open(f'{file_path}/output.csv', 'w')
for row in csv_rows:
    row_str = ','.join(map(str, row))
    outfile.write(row_str + '\n')
outfile.close()
print('Done.')