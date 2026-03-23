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

    extractor = FeatureExtractor()
    brain = IDSModel()
    
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
                # Subtract 8.5 hours (30,600 seconds) to align IST with Canada Time
                pkt_time_float = float(pkt.time) - 30600
                current_time = datetime.fromtimestamp(pkt_time_float).strftime('%H:%M:%S')    
                
                # Create the legally accurate ISO timestamp for the XAI JSON Logger
                forensic_timestamp = datetime.fromtimestamp(pkt_time_float).isoformat() + "Z"

                # --- TIME TRAVEL SKIP ---
                if pkt_time_float < START_TIMESTAMP:
                    # Update the live clock every 100,000 packets during the fast skip
                    if i % 100000 == 0:
                        sys.stdout.write(f"\r[ |>>>| FAST-FORWARD] Skimming morning traffic... Current PCAP Time: {current_time} | Packets Skipped: {i}")
                        sys.stdout.flush()
                    continue

                # --- LIVE UI STATUS BAR (Updates every 1000 packets to prevent lag) ---
                if i % 1000 == 0:
                    sys.stdout.write(f"\r[ |#| LIVE ANALYSIS] PCAP Time: {current_time} | Packets: {i} | Attacks: {attack_count}    ")
                    sys.stdout.flush()

                # --- EXTRACT & PREDICT ---
                features = extractor.extract_features(pkt)
                
                if features:
                    src_ip = pkt[IP].src if IP in pkt else "Unknown"
                    dst_ip = pkt[IP].dst if IP in pkt else "Unknown"
                    
                    # Pass the historical forensic timestamp into the AI Engine
                    verdict, confidence, xai_data = brain.predict(
                        features, src_ip=src_ip, dst_ip=dst_ip, pkt_time=forensic_timestamp
                    )
                    
                    # --- VERIFICATION LOGGING ---
                    if "ATTACK" in verdict:
                        attack_count += 1
                        # We print a \n first so it doesn't overwrite our live clock, 
                        # pushing the alert up and keeping the clock at the bottom!
                        print(f"\n[{current_time}] <!!!> ATTACK | Conf: {confidence:.1f}% | {src_ip} -> {dst_ip} | Port: {features[' Destination Port']}")
                    else:
                        normal_count += 1
                        # Print a sample of NORMAL traffic to prove the AI isn't just blindly guessing
                        if normal_count % 10000 == 0:
                            print(f"\n[{current_time}] <-+-> NORMAL | Conf: {confidence:.1f}% | {src_ip} -> {dst_ip} | Port: {features[' Destination Port']}")

    except KeyboardInterrupt:
        print("\n\n[!] Analysis stopped by user.")
    except Exception as e:
        print(f"\n\n[-] Error: {e}")

    print("\n" + "-" * 85)
    print(f"[+] PCAP Analysis Complete.")
    print(f"[+] Total Attacks Detected: {attack_count}")
    print(f"[+] Total Normal Flows Processed (Post-Skip): {normal_count}")

if __name__ == "__main__":
    pcap_file = "data/Friday-WorkingHours.pcap"  
    analyze_pcap(pcap_file)