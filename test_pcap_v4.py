import os
import sys
from datetime import datetime
import pandas as pd
from scapy.utils import PcapReader
from scapy.layers.inet import IP

from core.extractor_v4 import FeatureExtractorV4
from core.ids_model_v4 import IDSModelV4
from core.explainer_v4 import XAIEngineV4

def analyze_pcap(pcap_path):
    print(f"[*] Loading Network Traffic from: {pcap_path}")
    if not os.path.exists(pcap_path):
        print("[-] PCAP file not found. Please provide a valid .pcap file.")
        return

    # --- DYNAMIC LOGGING UPDATE (V4 Sandbox) ---
    base_name = os.path.splitext(os.path.basename(pcap_path))[0]
    # Must end in "v4.jsonl" so app_v4.py automatically picks it up!
    dynamic_log_name = f"{base_name}_v4.jsonl" 
    
    extractor = FeatureExtractorV4()
    
    # Boot the V4 Brain and the separate V4 SHAP Explainer
    brain = IDSModelV4() 
    explainer = XAIEngineV4(brain.model, brain.features, log_filename=dynamic_log_name)
    
    # 1. TIME TRAVEL TARGET: 13:45:00 (10 minutes before Port Scans begin)
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
                    sys.stdout.write(f"\r[ |#| LIVE ANALYSIS (V4)] PCAP Time: {current_time} | Packets: {i} | Attacks: {attack_count}    ")
                    sys.stdout.flush()

                # --- EXTRACT & PREDICT ---
                features = extractor.extract_features(pkt)
                
                if features:
                    src_ip = pkt[IP].src if IP in pkt else "Unknown"
                    dst_ip = pkt[IP].dst if IP in pkt else "Unknown"
                    
                    # V4 Prediction
                    verdict, confidence, metadata = brain.predict(
                        features, src_ip=src_ip, dst_ip=dst_ip, pkt_time=forensic_timestamp
                    )
                    
                    # --- VERIFICATION & LOGGING ---
                    if verdict is True:
                        attack_count += 1
                        port = metadata.get('port', 'N/A') if metadata else features.get(' Destination Port', 'N/A')
                        print(f"\n[{current_time}] <!!!> ATTACK | Conf: {confidence:.1f}% | {src_ip} -> {dst_ip} | Port: {port}")
                        
                        # V4 Explainer requires a Pandas DataFrame
                        try:
                            df_features = pd.DataFrame([features], columns=brain.features)
                            df_features.fillna(0, inplace=True) # Fills missing Claude features with 0
                            explainer.log_detection(df_features, confidence, port, src_ip, dst_ip, forensic_timestamp)
                        except Exception as e:
                            print(f"\n[-] SHAP Logging Error: {e}")
                            pass
                            
                    else:
                        normal_count += 1
                        if normal_count % 10000 == 0:
                            port = features.get(' Destination Port', 'N/A')
                            print(f"\n[{current_time}] <-+-> NORMAL | Conf: {confidence:.1f}% | {src_ip} -> {dst_ip} | Port: {port}")

    except KeyboardInterrupt:
        print("\n\n[!] Analysis stopped by user.")
    except Exception as e:
        print(f"\n\n[-] Error: {e}")

    print("\n" + "-" * 85)
    print(f"[+] V4 PCAP Analysis Complete.")
    print(f"[+] Total Attacks Detected: {attack_count}")
    print(f"[+] Total Normal Flows Processed: {normal_count}")

if __name__ == "__main__":
    #pcap_file = "data/bigNormal.pcapng"  # PCAP for testing no attacks
    pcap_file = "data/PortScanPcapFridayWorkingHrs.pcapng"  # PCAP with port scans
    analyze_pcap(pcap_file)