import os
from scapy.all import rdpcap
from core.extractor import FeatureExtractor
from core.ids_model import IDSModel

# We will use your training CSV as a mock test for now if you don't have a PCAP.
# Ideally, you will download a small sample PCAP file for this.
def analyze_pcap(pcap_path):
    print(f"[*] Loading Network Traffic from: {pcap_path}")
    if not os.path.exists(pcap_path):
        print("[-] PCAP file not found. Please provide a valid .pcap file.")
        return

    # 1. Initialize your custom modules
    extractor = FeatureExtractor()
    brain = IDSModel()
    
    print("[*] Reading packets (This might take a moment)...")
    packets = rdpcap(pcap_path)
    
    print(f"[*] Analyzing {len(packets)} packets...\n")
    print("-" * 50)
    
    # 2. The Core Pipeline Loop
    for i, pkt in enumerate(packets):
        # Translate raw packet to math
        features = extractor.extract_features(pkt)
        
        if features:
            # Feed math to AI
            verdict, confidence = brain.predict(features)
            
            # Print an alert if it's an attack, or just a status for normal traffic
            if "ATTACK" in verdict:
                print(f"[Packet {i}] 🚨 {verdict} | Confidence: {confidence:.2f}% | Dest Port: {features[' Destination Port']}")

    print("-" * 50)
    print("[+] PCAP Analysis Complete.")

if __name__ == "__main__":
    # You will need to place a sample .pcap file in your data folder
    analyze_pcap("data/sample_scan.pcap")