#!/usr/bin/env python3
"""
Diagnostic tool to examine raw audio data from ESP32
Shows statistics about the incoming data to help identify issues
"""

import serial
import serial.tools.list_ports
import numpy as np
import sys
import time

SAMPLE_RATE = 16000
NUM_MICS = 7
CHUNK_SIZE = 256

def diagnose(port, duration_sec=5):
    """Examine raw data from ESP32"""
    
    print("="*60)
    print("ESP32 Audio Diagnostics")
    print("="*60)
    print(f"Port: {port}")
    print(f"Duration: {duration_sec} seconds")
    print("="*60)
    
    try:
        ser = serial.Serial(port, 921600, timeout=2)
        print("✓ Serial connection opened\n")
    except Exception as e:
        print(f"✗ Failed to open {port}: {e}")
        return False
    
    time.sleep(3)
    ser.reset_input_buffer()
    
    bytes_per_chunk = CHUNK_SIZE * NUM_MICS * 2
    print(f"Expected bytes per chunk: {bytes_per_chunk}")
    print(f"Reading first 10 chunks to analyze...\n")
    
    all_chunks = []
    start_time = time.time()
    
    try:
        chunk_count = 0
        while time.time() - start_time < duration_sec and chunk_count < 10:
            chunk_bytes = ser.read(bytes_per_chunk)
            
            if len(chunk_bytes) == 0:
                print("⚠️  No data received - ESP32 may not be sending")
                continue
            
            if len(chunk_bytes) < bytes_per_chunk:
                print(f"⚠️  Incomplete chunk: {len(chunk_bytes)}/{bytes_per_chunk} bytes")
                continue
            
            # Convert to int16
            chunk = np.frombuffer(chunk_bytes, dtype=np.int16)
            all_chunks.append(chunk)
            
            # Show statistics for this chunk
            chunk_reshaped = chunk.reshape((CHUNK_SIZE, NUM_MICS))
            
            # Check each microphone
            for mic in range(NUM_MICS):
                mic_data = chunk_reshaped[:, mic]
                min_val = np.min(mic_data)
                max_val = np.max(mic_data)
                mean_val = np.mean(mic_data)
                std_val = np.std(mic_data)
                
                status = "📊" if std_val > 100 else "🔇"
                print(f"  Chunk {chunk_count} Mic {mic}: {status} "
                      f"min={min_val:6d} max={max_val:6d} "
                      f"mean={mean_val:7.1f} std={std_val:7.1f}")
            
            chunk_count += 1
            print()
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    ser.close()
    
    if not all_chunks:
        print("\n✗ No valid data received")
        return False
    
    # Overall statistics
    print("\n" + "="*60)
    print("OVERALL STATISTICS")
    print("="*60)
    all_data = np.concatenate(all_chunks)
    
    print(f"Total samples: {len(all_data):,}")
    print(f"Min value: {np.min(all_data)}")
    print(f"Max value: {np.max(all_data)}")
    print(f"Mean: {np.mean(all_data):.1f}")
    print(f"Std Dev: {np.std(all_data):.1f}")
    
    # Check if it's mostly quiet or mostly loud
    if np.std(all_data) < 100:
        print("\n⚠️  PROBLEM: Very low variation in data (static/white noise)")
        print("   Possible causes:")
        print("   1. Microphone not connected or not working")
        print("   2. Microphone gain is set to 0")
        print("   3. Wrong audio format or byte order")
    elif np.std(all_data) > 10000:
        print("\n⚠️  WARNING: Very high variation (possible clipping or noise)")
    else:
        print("\n✓ Data variation looks reasonable")
    
    return True

def main():
    if len(sys.argv) < 2:
        print("\nUsage: python diagnose_audio.py <PORT> [DURATION]")
        print("\nExamples:")
        print("  python diagnose_audio.py COM3")
        print("  python diagnose_audio.py COM3 5")
        print("\nAvailable ports:")
        
        ports = list(serial.tools.list_ports.comports())
        if ports:
            for p in ports:
                print(f"  {p.device}: {p.description}")
        else:
            print("  No serial ports found!")
        
        sys.exit(1)
    
    port = sys.argv[1]
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    diagnose(port, duration)

if __name__ == "__main__":
    main()
