from scapy.all import IP, TCP, UDP


class FeatureExtractorV4:
    """
    Feature extractor aligned with train_enhanced_v4.py's FEATURES_ENHANCED (16 features).

    Dropped from v3:
        SYN Flag Count, RST Flag Count, ACK Flag Count, FIN Flag Count

    Added in v4:
        Flow IAT Max          – max inter-arrival time across all packets
        Flow Bytes/s          – total bytes transferred per second
        Fwd Packets/s         – forward packets per second
        Bwd Packets/s         – backward packets per second
        Packet Length Variance– variance of per-packet lengths in the flow
        Init_Win_bytes_forward– initial TCP receive-window advertised by the client
    """

    def __init__(self):
        # Tracks ongoing conversations (flows) between IP pairs
        self.active_flows = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def get_flow_key(self, packet):
        """
        Returns (flow_key, direction) for a TCP/UDP packet, or None for others.
        Direction is "forward" for the initiating side, "backward" for the reply.
        """
        if IP not in packet:
            return None

        src_ip   = packet[IP].src
        dst_ip   = packet[IP].dst
        proto    = packet[IP].proto

        if TCP in packet:
            src_port = packet[TCP].sport
            dst_port = packet[TCP].dport
        elif UDP in packet:
            src_port = packet[UDP].sport
            dst_port = packet[UDP].dport
        else:
            return None  # Ignore ICMP and other non-TCP/UDP

        forward_key  = (src_ip, dst_ip, src_port, dst_port, proto)
        backward_key = (dst_ip, src_ip, dst_port, src_port, proto)

        if backward_key in self.active_flows:
            return backward_key, "backward"
        return forward_key, "forward"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_features(self, packet):
        """
        Accepts a raw Scapy packet, updates flow state, and returns the 16
        features expected by cat_enhanced_v4.cbm. Returns None for packets
        that cannot be classified (non-IP, non-TCP/UDP, etc.).
        """
        flow_info = self.get_flow_key(packet)
        if not flow_info:
            return None

        key, direction = flow_info
        timestamp   = float(packet.time)
        payload_len = len(packet)

        # ── 1. Initialise a new flow record ───────────────────────────
        if key not in self.active_flows:
            # Capture the initial TCP window size from the very first
            # forward (SYN) packet – this is Init_Win_bytes_forward.
            init_win = 0
            if TCP in packet and direction == "forward":
                init_win = packet[TCP].window

            self.active_flows[key] = {
                'first_time':      timestamp,
                'last_time':       timestamp,
                'prev_time':       timestamp,   # used to compute per-packet IAT
                'fwd_pkts':        0,
                'bwd_pkts':        0,
                'fwd_len':         0,
                'bwd_len':         0,
                'fwd_max':         0,
                'bwd_max':         0,
                'dst_port':        key[3],      # index 3 in (src_ip,dst_ip,sport,dport,proto)
                # Inter-arrival time tracking
                'iat_values':      [],           # list of all IAT values (seconds)
                'iat_max':         0.0,
                # Packet length list for variance computation
                'pkt_lengths':     [],
                # Init TCP window (forward direction only, captured once)
                'init_win_fwd':    init_win,
            }

        flow = self.active_flows[key]

        # ── 2. Per-packet IAT update ──────────────────────────────────
        # IAT is computed between consecutive packets regardless of direction
        if flow['fwd_pkts'] + flow['bwd_pkts'] > 0:
            iat = timestamp - flow['prev_time']
            flow['iat_values'].append(iat)
            if iat > flow['iat_max']:
                flow['iat_max'] = iat
        flow['prev_time'] = timestamp

        # ── 3. Packet length accumulator (for variance) ───────────────
        flow['pkt_lengths'].append(payload_len)

        # ── 4. Direction-specific counters ───────────────────────────
        if direction == "forward":
            flow['fwd_pkts'] += 1
            flow['fwd_len']  += payload_len
            if payload_len > flow['fwd_max']:
                flow['fwd_max'] = payload_len
        else:
            flow['bwd_pkts'] += 1
            flow['bwd_len']  += payload_len
            if payload_len > flow['bwd_max']:
                flow['bwd_max'] = payload_len

        # ── 5. Capture Init_Win_bytes_forward once (first SYN) ───────
        # If somehow the flow was created by a backward packet first,
        # update when we finally see the true forward direction.
        if (TCP in packet
                and direction == "forward"
                and flow['init_win_fwd'] == 0):
            flow['init_win_fwd'] = packet[TCP].window

        # ── 6. Time & rate calculations ──────────────────────────────
        flow['last_time'] = timestamp

        # CIC-IDS2017 expresses duration in microseconds
        duration_micro = (flow['last_time'] - flow['first_time']) * 1e6
        duration_sec   = duration_micro / 1e6

        total_pkts  = flow['fwd_pkts'] + flow['bwd_pkts']
        total_bytes = flow['fwd_len']  + flow['bwd_len']

        # Guards against division-by-zero
        safe_dur_sec       = max(duration_sec, 1e-6)
        safe_pkts_minus_1  = max(total_pkts - 1, 1)

        # IAT Mean (microseconds, consistent with Flow Duration units)
        iat_mean_micro = duration_micro / safe_pkts_minus_1
        # IAT Max (microseconds)
        iat_max_micro  = flow['iat_max'] * 1e6

        # Packet Length Variance
        pkt_var = float(np.var(flow['pkt_lengths'])) if len(flow['pkt_lengths']) > 1 else 0.0

        # ── 7. Assemble the 16-feature dictionary ────────────────────
        # Feature names and spacing must exactly match FEATURES_ENHANCED
        # in train_enhanced_v4.py so the model receives the correct columns.
        features = {
            ' Destination Port':            float(flow['dst_port']),
            ' Flow Duration':               float(duration_micro),
            ' Total Fwd Packets':           float(flow['fwd_pkts']),
            ' Total Bwd Packets':           float(flow['bwd_pkts']),
            'Total Length of Fwd Packets':  float(flow['fwd_len']),
            ' Total Length of Bwd Packets': float(flow['bwd_len']),
            ' Fwd Packet Length Max':       float(flow['fwd_max']),
            ' Bwd Packet Length Max':       float(flow['bwd_max']),
            ' Flow IAT Mean':               float(iat_mean_micro),
            ' Flow IAT Max':                float(iat_max_micro),
            ' Flow Packets/s':              float(total_pkts  / safe_dur_sec),
            ' Flow Bytes/s':                float(total_bytes / safe_dur_sec),
            ' Fwd Packets/s':               float(flow['fwd_pkts'] / safe_dur_sec),
            ' Bwd Packets/s':               float(flow['bwd_pkts'] / safe_dur_sec),
            ' Packet Length Variance':      float(pkt_var),
            'Init_Win_bytes_forward':       float(flow['init_win_fwd']),
        }

        return features

    def clear_flow(self, packet):
        """
        Removes a completed/reset flow from memory.
        Call this when a FIN or RST is observed to free up space.
        """
        flow_info = self.get_flow_key(packet)
        if flow_info:
            key, _ = flow_info
            self.active_flows.pop(key, None)

    def active_flow_count(self):
        """Returns the number of flows currently being tracked."""
        return len(self.active_flows)


# ---------------------------------------------------------------------------
# numpy is needed for variance – imported at module level for clarity
# ---------------------------------------------------------------------------
import numpy as np  # noqa: E402  (placed after class so the docstring reads first)
