"""
utils/flow_extractor.py
Extracts network flows from PCAP files using CICFlowMeter.
Self-contained implementation - no external dependencies on live_flow_extractor.py
"""

import pandas as pd
from pathlib import Path

# Import CICFlowMeter with proper error handling
try:
    from scapy.all import sniff
    from cicflowmeter.flow_session import FlowSession
except ImportError as e:
    raise ImportError(
        f"Required dependencies not found: {e}\n"
        "Please install: pip install scapy git+https://github.com/hieulw/cicflowmeter.git@master"
    )


def extract_flows_from_pcap(input_pcap, output_csv):
    """
    Extract flows from PCAP file using CICFlowMeter.
    
    Args:
        input_pcap: Path to input PCAP/PCAPNG file
        output_csv: Path to output CSV file with extracted flows
        
    Raises:
        FileNotFoundError: If PCAP file doesn't exist
        RuntimeError: If flow extraction fails
        ImportError: If CICFlowMeter dependencies are not available
    """
    # Convert to Path objects and resolve absolute paths
    input_pcap = Path(input_pcap).resolve()
    output_csv = Path(output_csv).resolve()
    
    # Validate input file exists
    if not input_pcap.exists():
        raise FileNotFoundError(f"PCAP file not found: {input_pcap}")
    
    # Ensure output directory exists
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"[+] Input PCAP: {input_pcap}")
    print(f"[+] Output CSV: {output_csv}")
    
    # Initialize CICFlowMeter session
    try:
        session = FlowSession(output_mode="csv", output=str(output_csv))
        print("[+] FlowSession initialized successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize CICFlowMeter FlowSession: {e}")
    
    # Process packets
    print("[+] Reading packets and extracting flows...")
    try:
        sniff(offline=str(input_pcap), prn=session.process, store=0)
    except Exception as e:
        raise RuntimeError(f"Error during packet processing: {e}")
    
    # Flush remaining flows to CSV
    print("[+] Flushing active flows to CSV...")
    if hasattr(session, 'flush_flows'):
        try:
            session.flush_flows()
        except Exception as e:
            print(f"[!] Warning: Error flushing flows: {e}")
    else:
        print("[!] Warning: 'flush_flows' method not found in FlowSession")
    
    # Verify output was created
    if not output_csv.exists():
        raise RuntimeError("Output CSV was not created after flow extraction")
    
    if output_csv.stat().st_size == 0:
        raise RuntimeError("CSV file is empty! (No TCP/UDP flows found in PCAP)")
    
    # Verify CSV is readable and show summary
    try:
        df = pd.read_csv(output_csv, low_memory=False)
        print("-" * 50)
        print(f"✓ SUCCESS!")
        print(f"✓ Total Flows Extracted: {len(df)}")
        print(f"✓ Total Features: {len(df.columns)}")
        print("-" * 50)
        print(f"Features Preview: {df.columns[:5].tolist()}")
    except Exception as e:
        raise RuntimeError(f"CSV created but unreadable: {e}")
