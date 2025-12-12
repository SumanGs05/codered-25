#!/usr/bin/env python3
"""
Simple audio playback test
Plays microphone channel 0 directly through speakers
No beamforming - just verify you can hear audio
"""

import serial
import struct
import numpy as np
import pyaudio
import time
import sys

SERIAL_PORT = '/dev/ttyACM0'
SERIAL_BAUD = 2000000
SAMPLE_RATE = 16000
CHANNELS = 7
CHUNK_SIZE = 512

class RobustAudioReceiver:
    def __init__(self, port, baud):
        self.ser = serial.Serial(port, baud, timeout=0.1)
        self.sync_buffer = bytearray()
        
    def find_mic_header(self):
        while True:
            byte = self.ser.read(1)
            if not byte:
                return False
            self.sync_buffer.append(byte[0])
            if len(self.sync_buffer) > 3:
                self.sync_buffer.pop(0)
            if len(self.sync_buffer) == 3 and bytes(self.sync_buffer) == b'MIC':
                self.sync_buffer.clear()
                return True
    
    def read_packet(self):
        if not self.find_mic_header():
            return None
        
        header_rest = self.ser.read(3)
        if len(header_rest) != 3:
            return None
        
        n_samples = struct.unpack('<H', header_rest[:2])[0]
        n_channels = header_rest[2]
        
        if n_samples > 2048 or n_channels != CHANNELS:
            return None
        
        data_size = n_samples * n_channels * 2
        audio_bytes = self.ser.read(data_size)
        
        if len(audio_bytes) != data_size:
            return None
        
        try:
            audio_int = np.frombuffer(audio_bytes, dtype=np.int16)
            return audio_int.reshape(n_samples, n_channels)
        except:
            return None

def main():
    print("=" * 60)
    print("Simple Audio Playback Test")
    print("=" * 60)
    
    # Connect to ESP32
    try:
        receiver = RobustAudioReceiver(SERIAL_PORT, SERIAL_BAUD)
        print(f"\n✓ Connected to {SERIAL_PORT}")
        time.sleep(1)
        
        # Clear startup messages
        for _ in range(10):
            line = receiver.ser.readline()
            if b'START' in line:
                break
        time.sleep(0.5)
        receiver.ser.reset_input_buffer()
        
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
    
    # Setup audio output
    print("\nSetting up audio output...")
    p = pyaudio.PyAudio()
    
    print("\nAvailable audio devices:")
    default_output = p.get_default_output_device_info()
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxOutputChannels'] > 0:
            marker = " [DEFAULT]" if i == default_output['index'] else ""
            print(f"  [{i}] {info['name']}{marker}")
    
    try:
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            output=True,
            frames_per_buffer=CHUNK_SIZE
        )
    except Exception as e:
        print(f"\nERROR: Could not open audio output")
        print(f"  {e}")
        print("\nTry:")
        print("  1. Check speakers/headphones are connected")
        print("  2. Run: aplay -l")
        print("  3. Adjust volume: alsamixer")
        sys.exit(1)
    
    print(f"\n✓ Audio output ready: {default_output['name']}")
    print(f"✓ Sample rate: {SAMPLE_RATE} Hz")
    print(f"✓ Latency: ~{CHUNK_SIZE/SAMPLE_RATE*1000:.0f} ms")
    
    print("\n" + "=" * 60)
    print("PLAYING MICROPHONE 0 (Left channel)")
    print("=" * 60)
    print("\nSpeak into the microphone array!")
    print("You should hear yourself with a slight delay.")
    print("\nPress Ctrl+C to stop\n")
    
    packets = 0
    errors = 0
    start_time = time.time()
    last_stats = time.time()
    
    try:
        while True:
            audio_data = receiver.read_packet()
            
            if audio_data is None:
                errors += 1
                continue
            
            packets += 1
            
            # Extract channel 0 (left mic from MIC_D0)
            mono = audio_data[:, 0]
            
            # Optional: Apply gain (uncomment if too quiet)
            # mono = np.clip(mono * 2, -32768, 32767).astype(np.int16)
            
            # Play audio
            stream.write(mono.tobytes())
            
            # Print stats every 2 seconds
            if time.time() - last_stats > 2.0:
                elapsed = time.time() - start_time
                audio_float = mono.astype(np.float32) / 32768.0
                rms = np.sqrt(np.mean(audio_float**2))
                peak = np.max(np.abs(audio_float))
                
                print(f"\r[{packets:4d} pkts | {packets/elapsed:.1f}/s] "
                      f"RMS: {rms:.3f} | Peak: {peak:.3f} | "
                      f"Errors: {errors}", 
                      end='', flush=True)
                
                last_stats = time.time()
    
    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("STOPPED")
        print("=" * 60)
        
        elapsed = time.time() - start_time
        print(f"\nStatistics:")
        print(f"  Duration:     {elapsed:.1f} seconds")
        print(f"  Packets:      {packets}")
        print(f"  Errors:       {errors}")
        print(f"  Success rate: {packets/(packets+errors)*100:.1f}%")
        
        print("\n" + "=" * 60)
        
        if packets > 100:
            print("✓ Audio playback working!")
            print("\nDid you hear audio?")
            print("  YES → Ready for beamforming! Run: python3 beamformer.py")
            print("  NO  → Check speaker volume (alsamixer) or connection")
        else:
            print("⚠ Too few packets received")
            print("  Check ESP32 connection and restart it")
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    receiver.ser.close()

if __name__ == '__main__':
    main()