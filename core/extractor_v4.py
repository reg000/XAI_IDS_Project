"""
extractor_v4.py  –  V4 Feature Extractor (fixed)
=================================================
Fix applied: rate features (Flow Packets/s, Flow Bytes/s, Fwd/Bwd Packets/s)
are set to 0.0 when the flow has only a single packet or zero real duration.

Why this matters:
  CICFlowMeter records rate = inf for zero-duration flows and those rows get
  removed by the training script's dropna().  The model therefore never saw
  values like 1_000_000 packets/s during training, making every live
  single-packet flow completely out-of-distribution → P(attack) < 4%.

  By mirroring CICFlowMeter's behaviour (treat single-packet flows as
  having rate = 0 rather than 1/epsilon), we restore the feature
  distribution the model was trained on.
"""

import numpy as np
from scapy.all import IP, TCP, UDP


class FeatureExtractorV4:
    """
    Extracts the 16-feature set aligned with train_enhanced_v4.py's
    FEATURES_ENHANCED list and the cat_enhanced_v4.cbm model.

    Feature set (16):
        1.  Destination Port
        2.  Flow Duration (µs)
        3.  Total Fwd Packets
        4.  Total Bwd Packets
        5.  Total Length of Fwd Packets
        6.  Total Length of Bwd Packets
        7.  Fwd Packet Length Max
        8.  Bwd Packet Length Max
        9.  Flow IAT Mean (µs)
        10. Flow IAT Max (µs)
        11. Flow Packets/s          <- 0 when single-packet / zero-duration flow
        12. Flow Bytes/s            <- 0 when single-packet / zero-duration flow
        13. Fwd Packets/s           <- 0 when single-packet / zero-duration flow
        14. Bwd Packets/s           <- 0 when single-packet / zero-duration flow
        15. Packet Length Variance
        16. Init_Win_bytes_forward
    """

    def __init__(self):
        self.active_flows: dict = {}

    # ── helpers ────────────────────────────────────────────────────────────

    def get_flow_key(self, packet):
        """
        Returns (flow_key, direction) or None for non-TCP/UDP packets.
        Bidirectional flows are keyed by the initiating direction.
        """
        if IP not in packet:
            return None

        src_ip = packet[IP].src
        dst_ip = packet[IP].dst
        proto  = packet[IP].proto

        if TCP in packet:
            src_port = packet[TCP].sport
            dst_port = packet[TCP].dport
        elif UDP in packet:
            src_port = packet[UDP].sport
            dst_port = packet[UDP].dport
        else:
            return None  # drop ICMP etc.

        fwd_key = (src_ip, dst_ip, src_port, dst_port, proto)
        bwd_key = (dst_ip, src_ip, dst_port, src_port, proto)

        if bwd_key in self.active_flows:
            return bwd_key, "backward"
        return fwd_key, "forward"

    # ── public API ─────────────────────────────────────────────────────────

    def extract_features(self, packet):
        """
        Update flow state for *packet* and return a 16-feature dict,
        or None if the packet cannot be classified.
        """
        flow_info = self.get_flow_key(packet)
        if not flow_info:
            return None

        key, direction = flow_info
        timestamp   = float(packet.time)
        payload_len = len(packet)

        # ── initialise new flow ────────────────────────────────────────
        if key not in self.active_flows:
            init_win = 0
            if TCP in packet and direction == "forward":
                init_win = packet[TCP].window

            self.active_flows[key] = {
                'first_time':   timestamp,
                'last_time':    timestamp,
                'prev_time':    timestamp,
                'fwd_pkts':     0,
                'bwd_pkts':     0,
                'fwd_len':      0,
                'bwd_len':      0,
                'fwd_max':      0,
                'bwd_max':      0,
                'dst_port':     key[3],
                'iat_max':      0.0,
                'pkt_lengths':  [],
                'init_win_fwd': init_win,
            }

        flow = self.active_flows[key]

        # ── IAT update (skip on first packet of a flow) ────────────────
        total_before = flow['fwd_pkts'] + flow['bwd_pkts']
        if total_before > 0:
            iat = timestamp - flow['prev_time']
            if iat > flow['iat_max']:
                flow['iat_max'] = iat
        flow['prev_time'] = timestamp

        # ── packet length accumulator ──────────────────────────────────
        flow['pkt_lengths'].append(payload_len)

        # ── directional counters ───────────────────────────────────────
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

        # ── capture initial TCP window (forward direction, once only) ──
        if TCP in packet and direction == "forward" and flow['init_win_fwd'] == 0:
            flow['init_win_fwd'] = packet[TCP].window

        # ── time calculations ──────────────────────────────────────────
        flow['last_time'] = timestamp

        duration_sec   = flow['last_time'] - flow['first_time']
        duration_micro = duration_sec * 1e6

        total_pkts  = flow['fwd_pkts'] + flow['bwd_pkts']
        total_bytes = flow['fwd_len']  + flow['bwd_len']

        # IAT mean in microseconds (guard against single-packet divide-by-zero)
        safe_pkts_minus_1 = max(total_pkts - 1, 1)
        iat_mean_micro    = duration_micro / safe_pkts_minus_1
        iat_max_micro     = flow['iat_max'] * 1e6

        # Packet length variance (meaningful only with 2+ packets)
        pkt_lengths = flow['pkt_lengths']
        pkt_var = float(np.var(pkt_lengths)) if len(pkt_lengths) > 1 else 0.0

        # ── rate features (KEY FIX) ────────────────────────────────────
        # CICFlowMeter emits inf when duration == 0; the training pipeline
        # removes those rows via dropna().  The model was therefore NEVER
        # trained on values like 1_000_000 packets/s that the old extractor
        # produced via the 1e-6 epsilon guard.  We mirror CICFlowMeter by
        # emitting 0.0 for any zero-duration flow instead.
        if duration_sec > 0:
            flow_pkts_s  = total_pkts  / duration_sec
            flow_bytes_s = total_bytes / duration_sec
            fwd_pkts_s   = flow['fwd_pkts'] / duration_sec
            bwd_pkts_s   = flow['bwd_pkts'] / duration_sec
        else:
            flow_pkts_s  = 0.0
            flow_bytes_s = 0.0
            fwd_pkts_s   = 0.0
            bwd_pkts_s   = 0.0

        # ── assemble feature dict ──────────────────────────────────────
        # Names and spacing must exactly match FEATURES_ENHANCED in
        # train_enhanced_v4.py (and therefore the saved feature_names_v4.joblib).
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
            ' Flow Packets/s':              float(flow_pkts_s),
            ' Flow Bytes/s':                float(flow_bytes_s),
            ' Fwd Packets/s':               float(fwd_pkts_s),
            ' Bwd Packets/s':               float(bwd_pkts_s),
            ' Packet Length Variance':      float(pkt_var),
            'Init_Win_bytes_forward':       float(flow['init_win_fwd']),
        }

        return features

    # ── memory management ──────────────────────────────────────────────────

    def clear_flow(self, packet):
        """
        Remove a completed or reset flow from memory.
        Call when a FIN or RST packet is observed.
        """
        flow_info = self.get_flow_key(packet)
        if flow_info:
            key, _ = flow_info
            self.active_flows.pop(key, None)

    def active_flow_count(self) -> int:
        """Number of flows currently tracked."""
        return len(self.active_flows)
