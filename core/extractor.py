from scapy.all import IP, TCP, UDP

class FeatureExtractor:
    def __init__(self):
        # We use a dictionary to track ongoing "conversations" (Flows) between IPs
        self.active_flows = {}

    def get_flow_key(self, packet):
        """Creates a unique ID for a two-way network conversation."""
        if IP not in packet:
            return None
            
        src_ip = packet[IP].src
        dst_ip = packet[IP].dst
        proto = packet[IP].proto
        
        # Get ports if it's TCP or UDP
        if TCP in packet:
            src_port = packet[TCP].sport
            dst_port = packet[TCP].dport
        elif UDP in packet:
            src_port = packet[UDP].sport
            dst_port = packet[UDP].dport
        else:
            return None # Ignore non-TCP/UDP traffic like ICMP ping
            
        # To group bidirectional traffic, we sort the IPs and Ports
        # This ensures A->B and B->A update the same flow record
        forward_key = (src_ip, dst_ip, src_port, dst_port, proto)
        backward_key = (dst_ip, src_ip, dst_port, src_port, proto)
        
        if backward_key in self.active_flows:
            return backward_key, "backward"
        return forward_key, "forward"

    def extract_features(self, packet):
        """
        Takes a raw Scapy packet, updates the flow math, 
        and returns the 14 specific AI features (10 Original + 4 TCP Flags).
        """
        flow_info = self.get_flow_key(packet)
        if not flow_info:
            return None
            
        key, direction = flow_info
        timestamp = float(packet.time)
        payload_len = len(packet)

        # 1. Initialize a new flow if we haven't seen this conversation before
        if key not in self.active_flows:
            self.active_flows[key] = {
                'first_time': timestamp,
                'last_time': timestamp,
                'fwd_pkts': 0,
                'bwd_pkts': 0,
                'fwd_len': 0,
                'bwd_len': 0,
                'fwd_max': 0,
                'bwd_max': 0,
                'dst_port': key[3], # Destination Port is index 3 in our tuple
                # --- NEW: Initialize Flag Counters ---
                'syn_count': 0,
                'rst_count': 0,
                'ack_count': 0,
                'fin_count': 0
            }

        flow = self.active_flows[key]
        
        # 2. Update math based on direction
        if direction == "forward":
            flow['fwd_pkts'] += 1
            flow['fwd_len'] += payload_len
            flow['fwd_max'] = max(flow['fwd_max'], payload_len)
        else:
            flow['bwd_pkts'] += 1
            flow['bwd_len'] += payload_len
            flow['bwd_max'] = max(flow['bwd_max'], payload_len)

        # --- NEW: Count TCP Flags for the Flow ---
        if TCP in packet:
            flags = packet[TCP].flags
            if 'S' in flags: flow['syn_count'] += 1  # SYN (Start/Stealth)
            if 'R' in flags: flow['rst_count'] += 1  # RST (Reset/Blocked)
            if 'A' in flags: flow['ack_count'] += 1  # ACK (Acknowledge)
            if 'F' in flags: flow['fin_count'] += 1  # FIN (Finish)

        # 3. Time Calculations
        flow['last_time'] = timestamp
        # CIC-IDS2017 calculates duration in microseconds
        duration_micro = (flow['last_time'] - flow['first_time']) * 1e6 
        duration_sec = duration_micro / 1e6

        total_pkts = flow['fwd_pkts'] + flow['bwd_pkts']
        
        # Avoid division by zero errors
        safe_duration_sec = max(duration_sec, 0.000001) 
        safe_pkts_minus_one = max(total_pkts - 1, 1)

        # 4. Final Feature Dictionary (Notice the exact spacing to match CIC-IDS2017!)
        features = {
            # Original 10 Features
            ' Destination Port': float(flow['dst_port']),
            ' Flow Duration': float(duration_micro),
            ' Total Fwd Packets': float(flow['fwd_pkts']),
            ' Total Bwd Packets': float(flow['bwd_pkts']),
            'Total Length of Fwd Packets': float(flow['fwd_len']),
            ' Total Length of Bwd Packets': float(flow['bwd_len']),
            ' Fwd Packet Length Max': float(flow['fwd_max']),
            ' Bwd Packet Length Max': float(flow['bwd_max']),
            ' Flow IAT Mean': float(duration_micro / safe_pkts_minus_one),
            ' Flow Packets/s': float(total_pkts / safe_duration_sec),
            
            # --- NEW: The 4 TCP Flag Features ---
            ' SYN Flag Count': float(flow['syn_count']),
            ' RST Flag Count': float(flow['rst_count']),
            ' ACK Flag Count': float(flow['ack_count']),
            ' FIN Flag Count': float(flow['fin_count'])
        }
        
        return features