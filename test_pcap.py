import os
import sys
from datetime import datetime
from scapy.utils import PcapReader
from scapy.layers.inet import IP
from core.extractor import FeatureExtractor
from core.ids_model import IDSModel

def analyze_pcap(pcap_path):
    print(f"[*] Loading Network Traffic from: {pcap_path}")
    if not os.path.exists(pcap_path):
        print("[-] PCAP file not found. Please provide a valid .pcap file.")
        return

    # --- DYNAMIC LOGGING UPDATE ---
    # Extract the base name (e.g., "Friday-WorkingHours" from "data/Friday-WorkingHours.pcap")
    base_name = os.path.splitext(os.path.basename(pcap_path))[0]
    dynamic_log_name = f"{base_name}_alerts.jsonl"
    
    extractor = FeatureExtractor()
    
    # Boot the Brain and hand it the dynamic log name!
    # (Make sure your IDSModel's __init__ accepts log_filename as we discussed earlier)
    brain = IDSModel(log_filename=dynamic_log_name) 
    
    # 1. TIME TRAVEL TARGET: 13:50:00 (5 minutes before Port Scans begin)
    START_TIMESTAMP = datetime(2017, 7, 7, 13, 45, 0).timestamp()

    print("[*] Streaming packets (Press Ctrl+C to stop)...")
    print("-" * 85)
    
    normal_count = 0
    attack_count = 0

    try:
        with PcapReader(pcap_path) as pcap_reader:
            for i, pkt in enumerate(pcap_reader):
                                
                # --- TIME SYNC FIX ---
                pkt_time_float = float(pkt.time) - 30600
                current_time = datetime.fromtimestamp(pkt_time_float).strftime('%H:%M:%S')    
                forensic_timestamp = datetime.fromtimestamp(pkt_time_float).isoformat() + "Z"

                # --- TIME TRAVEL SKIP ---
                if pkt_time_float < START_TIMESTAMP:
                    if i % 100000 == 0:
                        sys.stdout.write(f"\r[ |>>>| FAST-FORWARD] Skimming morning traffic... Current PCAP Time: {current_time} | Packets Skipped: {i}")
                        sys.stdout.flush()
                    continue

                # --- LIVE UI STATUS BAR ---
                if i % 1000 == 0:
                    sys.stdout.write(f"\r[ |#| LIVE ANALYSIS] PCAP Time: {current_time} | Packets: {i} | Attacks: {attack_count}    ")
                    sys.stdout.flush()

                # --- EXTRACT & PREDICT ---
                features = extractor.extract_features(pkt)
                
                if features:
                    src_ip = pkt[IP].src if IP in pkt else "Unknown"
                    dst_ip = pkt[IP].dst if IP in pkt else "Unknown"
                    
                    # predict() will now write to your dynamically named JSONL!
                    verdict, confidence, xai_data = brain.predict(
                        features, src_ip=src_ip, dst_ip=dst_ip, pkt_time=forensic_timestamp
                    )
                    
                    # --- VERIFICATION LOGGING ---
                    if "ATTACK" in verdict or verdict is True: # Handles both tuple and boolean returns depending on your current ids_model state
                        attack_count += 1
                        print(f"\n[{current_time}] <!!!> ATTACK | Conf: {confidence:.1f}% | {src_ip} -> {dst_ip} | Port: {features.get(' Destination Port', 'N/A')}")
                    else:
                        normal_count += 1
                        if normal_count % 10000 == 0:
                            print(f"\n[{current_time}] <-+-> NORMAL | Conf: {confidence:.1f}% | {src_ip} -> {dst_ip} | Port: {features.get(' Destination Port', 'N/A')}")

    except KeyboardInterrupt:
        print("\n\n[!] Analysis stopped by user.")
    except Exception as e:
        print(f"\n\n[-] Error: {e}")

    print("\n" + "-" * 85)
    print(f"[+] PCAP Analysis Complete.")
    print(f"[+] Total Attacks Detected: {attack_count}")
    print(f"[+] Total Normal Flows Processed (Post-Skip): {normal_count}")

if __name__ == "__main__":
    # You can now easily swap this file out, and the logs will neatly separate!
    pcap_file = "data/Friday-WorkingHours.pcap"  
    analyze_pcap(pcap_file)