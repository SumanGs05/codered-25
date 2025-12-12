#!/usr/bin/env python3
"""
Simplified Audio Recorder - No handshake, just start recording
Usage: python simple_recorder.py COM3 10 output.wav
"""

import serial
import serial.tools.list_ports
import numpy as np
from scipy import signal
import wave
import sys
import time

SAMPLE_RATE = 16000
NUM_MICS = 7
CHUNK_SIZE = 256

def remove_dc_offset(audio):
    """Remove DC offset (mean value) from audio"""
    return audio - np.mean(audio)

def normalize_audio(audio, target_level=0.9):
    """Normalize audio to prevent clipping"""
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        return audio * (target_level / max_val)
    return audio

def apply_highpass_filter(audio, cutoff_hz=50):
    """Apply high-pass filter to remove low-frequency noise"""
    nyquist = SAMPLE_RATE / 2
    normalized_cutoff = cutoff_hz / nyquist
    
    # Design butterworth high-pass filter
    b, a = signal.butter(2, normalized_cutoff, btype='high')
    return signal.filtfilt(b, a, audio).astype(np.int16)

def record_audio(port, duration_sec, output_file):
    """Record audio from ESP32 without waiting for READY signal"""
    
    print("="*60)
    print("Simple ESP32 Audio Recorder")
    print("="*60)
    print(f"Port: {port}")
    print(f"Duration: {duration_sec} seconds")
    print(f"Output: {output_file}")
    print("="*60)
    
    # Open serial connection
    try:
        ser = serial.Serial(port, 921600, timeout=2)
        print("✓ Serial connection opened")
    except Exception as e:
        print(f"✗ Failed to open {port}: {e}")
        print("\nAvailable ports:")
        for p in serial.tools.list_ports.comports():
            print(f"  {p.device}: {p.description}")
        return False
    
    print("\n⏳ Waiting 3 seconds for ESP32 to initialize...")
    time.sleep(3)
    
    # Flush any startup data
    ser.reset_input_buffer()
    print("✓ Buffer cleared\n")
    
    # Calculate samples needed
    total_samples_needed = int(duration_sec * SAMPLE_RATE)
    bytes_per_chunk = CHUNK_SIZE * NUM_MICS * 2
    
    print(f"🔴 Recording {duration_sec} seconds...")
    print(f"Need {total_samples_needed:,} samples\n")
    
    audio_data = []
    samples_received = 0
    start_time = time.time()
    bytes_received = 0
    
    try:
        while samples_received < total_samples_needed:
            # Read one chunk
            chunk_bytes = ser.read(bytes_per_chunk)
            bytes_received += len(chunk_bytes)
            
            if len(chunk_bytes) < bytes_per_chunk:
                # Incomplete chunk - skip and continue
                continue
            
            # Convert to numpy array
            try:
                chunk = np.frombuffer(chunk_bytes, dtype=np.int16)
                chunk = chunk.reshape((CHUNK_SIZE, NUM_MICS))
                audio_data.append(chunk)
                samples_received += CHUNK_SIZE
            except:
                # Bad data - skip
                continue
            
            # Progress
            elapsed = time.time() - start_time
            progress = (samples_received / total_samples_needed) * 100
            actual_audio = samples_received / SAMPLE_RATE
            data_rate = bytes_received / elapsed / 1024
            
            print(f"\r Progress: {progress:5.1f}% | "
                  f"Audio: {actual_audio:5.1f}s / {duration_sec}s | "
                  f"Rate: {data_rate:6.1f} KB/s", 
                  end='', flush=True)
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopped by user")
    
    ser.close()
    
    # Process audio
    if not audio_data:
        print("\n\n✗ No valid audio data received")
        print("Check:")
        print("  1. ESP32 code is uploaded and running")
        print("  2. Serial Monitor in Arduino IDE is CLOSED")
        print("  3. Wiring is correct (especially GND and 5V)")
        return False
    
    # Concatenate chunks
    all_audio = np.vstack(audio_data)
    
    # Process each microphone channel individually
    print("\n\n📊 Processing audio...")
    processed_channels = []
    
    for mic in range(NUM_MICS):
        mic_data = all_audio[:, mic].astype(np.float32)
        
        # Remove DC offset
        mic_data = remove_dc_offset(mic_data)
        
        # Apply high-pass filter
        mic_data = apply_highpass_filter(mic_data)
        
        # Normalize
        mic_data = normalize_audio(mic_data)
        
        processed_channels.append(mic_data)
    
    # Mix to mono with proper scaling
    mono_audio = np.mean(processed_channels, axis=0).astype(np.int16)
    
    # Final normalization to prevent clipping
    mono_audio = normalize_audio(mono_audio).astype(np.int16)
    
    print(f"\n\n✓ Recording complete!")
    print(f"  Samples: {len(mono_audio):,}")
    print(f"  Duration: {len(mono_audio) / SAMPLE_RATE:.2f}s")
    print(f"  Data received: {bytes_received / 1024:.1f} KB")
    
    # Save to WAV
    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(mono_audio.tobytes())
    
    print(f"\n✅ Saved to: {output_file}")
    print("You can now play this file!\n")
    
    return True

def main():
    if len(sys.argv) < 4:
        print("\nUsage: python simple_recorder.py <PORT> <DURATION> <OUTPUT.wav>")
        print("\nExamples:")
        print("  python simple_recorder.py COM3 10 output.wav")
        print("  python simple_recorder.py /dev/ttyUSB0 5 test.wav")
        print("\nAvailable ports:")
        
        ports = list(serial.tools.list_ports.comports())
        if ports:
            for p in ports:
                print(f"  {p.device}: {p.description}")
        else:
            print("  No serial ports found!")
        
        sys.exit(1)
    
    port = sys.argv[1]
    duration = int(sys.argv[2])
    output = sys.argv[3]
    
    success = record_audio(port, duration, output)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()