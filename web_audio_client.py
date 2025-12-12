"""
ESP32 Web Audio Client
Fetches audio from ESP32 web server and plays through speakers
"""

import requests
import numpy as np
import pyaudio
import wave
import time
import sys

# Configuration
ESP32_IP = "10.112.187.149"  # Change this to your ESP32's IP
ESP32_PORT = 80
STREAM_URL = f"http://{ESP32_IP}:{ESP32_PORT}/stream"

SAMPLE_RATE = 16000
CHANNELS = 2  # Stereo
CHUNK_SIZE = 1024

class WebAudioClient:
    def __init__(self, esp32_url):
        self.url = esp32_url
        print("=" * 60)
        print("ESP32 Web Audio Client")
        print("=" * 60)
        print(f"Connecting to: {esp32_url}")
        
        # Test connection
        try:
            response = requests.get(f"http://{ESP32_IP}:{ESP32_PORT}/", timeout=5)
            if response.status_code == 200:
                print("✓ ESP32 server is reachable")
            else:
                print(f"⚠ Server responded with status {response.status_code}")
        except Exception as e:
            print(f"✗ Cannot reach ESP32: {e}")
            print("\nMake sure:")
            print("  1. ESP32 is powered on")
            print("  2. ESP32 is connected to WiFi")
            print("  3. IP address is correct")
            raise
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        
        print("\nAvailable audio devices:")
        for i in range(self.p.get_device_count()):
            info = self.p.get_device_info_by_index(i)
            if info['maxOutputChannels'] > 0:
                print(f"  [{i}] {info['name']}")
        
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            output=True,
            frames_per_buffer=CHUNK_SIZE
        )
        
        print("✓ Audio output initialized\n")
        
        # Statistics
        self.chunks_received = 0
        self.bytes_received = 0
        self.errors = 0
        self.last_stats_time = time.time()
    
    def fetch_audio_chunk(self):
        """Fetch one chunk of audio from ESP32"""
        try:
            response = requests.get(self.url, timeout=2)
            
            if response.status_code == 200:
                # Convert bytes to numpy array
                audio_data = np.frombuffer(response.content, dtype=np.int16)
                
                self.chunks_received += 1
                self.bytes_received += len(response.content)
                
                return audio_data
            else:
                self.errors += 1
                return None
                
        except requests.exceptions.Timeout:
            self.errors += 1
            print("⚠ Request timeout")
            return None
        except Exception as e:
            self.errors += 1
            print(f"⚠ Error fetching audio: {e}")
            return None
    
    def print_stats(self):
        """Print statistics"""
        current_time = time.time()
        elapsed = current_time - self.last_stats_time
        
        if elapsed >= 1.0:
            chunks_per_sec = self.chunks_received / elapsed
            bytes_per_sec = self.bytes_received / elapsed
            
            print(f"📊 Chunks/s: {chunks_per_sec:>6.1f} | "
                  f"Data: {bytes_per_sec/1024:>6.1f} KB/s | "
                  f"Errors: {self.errors:>4} | "
                  f"Total: {self.chunks_received:>6}")
            
            self.chunks_received = 0
            self.bytes_received = 0
            self.last_stats_time = current_time
    
    def stream_audio(self):
        """Stream audio in real-time"""
        print("🎤 Streaming audio from ESP32...")
        print("🔊 Playing through speakers")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                # Fetch audio chunk
                audio_data = self.fetch_audio_chunk()
                
                if audio_data is not None and len(audio_data) > 0:
                    # Play audio
                    self.stream.write(audio_data.tobytes())
                    
                    # Print stats
                    self.print_stats()
                else:
                    time.sleep(0.01)  # Small delay if no data
                    
        except KeyboardInterrupt:
            print("\n\n⏹️  Stopped by user")
        finally:
            self.cleanup()
    
    def record_to_file(self, duration_seconds=10, filename='recorded_audio.wav'):
        """Record audio to WAV file"""
        print(f"\n🔴 Recording {duration_seconds} seconds...")
        print(f"📁 Saving to: {filename}\n")
        
        frames = []
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration_seconds:
                audio_data = self.fetch_audio_chunk()
                
                if audio_data is not None and len(audio_data) > 0:
                    frames.append(audio_data)
                    
                    # Show progress
                    elapsed = time.time() - start_time
                    progress = (elapsed / duration_seconds) * 100
                    print(f"\rProgress: {progress:>5.1f}% | {elapsed:>5.1f}s / {duration_seconds}s", end='')
                
                time.sleep(0.01)
        
        except KeyboardInterrupt:
            print("\n\n⏹️  Recording stopped early")
        
        # Save to WAV file
        if frames:
            all_audio = np.concatenate(frames)
            
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(all_audio.tobytes())
            
            duration = len(all_audio) / (SAMPLE_RATE * CHANNELS)
            print(f"\n\n✅ Recording saved!")
            print(f"   Duration: {duration:.2f} seconds")
            print(f"   Samples: {len(all_audio):,}")
            print(f"   File: {filename}")
        else:
            print("\n\n❌ No audio data received")
        
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("\nCleaning up...")
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'p'):
            self.p.terminate()
        print("✓ Done")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python web_audio_client.py <ESP32_IP>              # Stream audio")
        print("  python web_audio_client.py <ESP32_IP> record [duration] [filename]")
        print("\nExamples:")
        print("  python web_audio_client.py 192.168.1.100")
        print("  python web_audio_client.py 192.168.1.100 record 10 test.wav")
        sys.exit(1)
    
    esp32_ip = sys.argv[1]
    stream_url = f"http://{esp32_ip}/stream"
    
    try:
        client = WebAudioClient(stream_url)
        
        if len(sys.argv) > 2 and sys.argv[2] == 'record':
            duration = int(sys.argv[3]) if len(sys.argv) > 3 else 10
            filename = sys.argv[4] if len(sys.argv) > 4 else 'recorded_audio.wav'
            client.record_to_file(duration, filename)
        else:
            client.stream_audio()
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()