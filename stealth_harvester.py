import pandas as pd
from scapy.all import PcapReader, IP, TCP
from datetime import datetime
import sys
import os

# Import your existing 14-feature Scapy extractor
from core.extractor import FeatureExtractor

PCAP_FILE = 'data/Friday-WorkingHours.pcap'
OUTPUT_CSV = 'data/stealth_augment.csv'

def harvest_stealth_scans():
    print(f"[*] Waking up Harvester Engine for {PCAP_FILE}...")
    extractor = FeatureExtractor()
    packet_count = 0
    
    # We are targeting the specific Port Scan windows identified in your log
    # Window: 13:50 (Pre-attack) to 15:30 (Post-Service Probing)
    with PcapReader(PCAP_FILE) as pcap:
        for pkt in pcap:
            packet_count += 1
            if IP not in pkt or TCP not in pkt:
                continue
                
            # Using raw PCAP time to match your analysis log
            pkt_time = datetime.fromtimestamp(float(pkt.time))
            
            # --- PHASE 1: FAST-FORWARD ---
            if pkt_time.hour < 13 or (pkt_time.hour == 13 and pkt_time.minute < 50):
                if packet_count % 500000 == 0:
                    sys.stdout.write(f"\r[⏩ FAST-FORWARD] Current PCAP Time: {pkt_time.strftime('%H:%M:%S')} | Packets: {packet_count}")
                continue
                
            # --- PHASE 2: HARVESTING WINDOW ---
            if pkt_time.hour >= 15 and pkt_time.minute > 30:
                print(f"\n[*] 15:30 reached. Extraction window closed.")
                break
                
            if packet_count % 10000 == 0:
                sys.stdout.write(f"\r[🔴 HARVESTING] Active Attack Window: {pkt_time.strftime('%H:%M:%S')} | Flows: {len(extractor.active_flows)}")
                
            extractor.extract_features(pkt)

    print("\n[*] Processing collected flows into the stealth training matrix...")
    extracted_data = []
    
    for key, flow in extractor.active_flows.items():
        # Capturing flows with SYN/RST activity typical of stealth reconnaissance
        if flow['syn_count'] > 0 or flow['rst_count'] > 0:
            duration_micro = (flow['last_time'] - flow['first_time']) * 1e6
            total_pkts = flow['fwd_pkts'] + flow['bwd_pkts']
            safe_duration_sec = max(duration_micro / 1e6, 0.000001)
            safe_pkts_minus_one = max(total_pkts - 1, 1)
            
            extracted_data.append({
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
                ' SYN Flag Count': float(flow['syn_count']),
                ' RST Flag Count': float(flow['rst_count']),
                ' ACK Flag Count': float(flow['ack_count']),
                ' FIN Flag Count': float(flow['fin_count']),
                ' Label': 'PortScan'
            })

    if extracted_data:
        df = pd.DataFrame(extracted_data)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"[+] Success! {len(df)} stealth-scan vectors saved to {OUTPUT_CSV}")
    else:
        print("[-] Error: No stealth flows found in this window. Check PCAP file path.")

if __name__ == "__main__":
    harvest_stealth_scans()