from flask import Flask, request, jsonify
import serial
import time
import json
import math

app = Flask(__name__)

current_state = {
    'zoom_level': 1.0,
    'beam_angle': 0.0,  
    'timestamp': time.time()
}

try:
    esp32_serial = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
    print("ESP32-S3 connected on /dev/ttyUSB0")
except:
    try:
        esp32_serial = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
        print("ESP32-S3 connected on /dev/ttyACM0")
    except:
        print(" ESP32-S3 not found. Check USB connection.")
        esp32_serial = None

def calculate_beam_angle(zoom_level):
    """
    Calculate beamforming angle based on camera zoom level.

    """
    base_fov = 60.0  
    current_fov = base_fov / zoom_level
    
    beam_width = current_fov
    
    return beam_width

def send_to_esp32(zoom_level, beam_angle):
    """Send beamforming parameters to ESP32-S3 via serial"""
    if esp32_serial is None:
        return False
    
    try:
        command = {
            'zoom': zoom_level,
            'angle': beam_angle,
            'timestamp': int(time.time() * 1000)
        }
        
        message = json.dumps(command) + '\n'
        esp32_serial.write(message.encode())
        esp32_serial.flush()
        
        return True
    except Exception as e:
        print(f"Error sending to ESP32: {e}")
        return False

@app.route('/zoom', methods=['POST'])
def receive_zoom():
    """Receive zoom level from phone"""
    data = request.get_json()
    
    if 'zoom_level' in data:
        zoom_level = data['zoom_level']
        beam_angle = calculate_beam_angle(zoom_level)
        
        current_state['zoom_level'] = zoom_level
        current_state['beam_angle'] = beam_angle
        current_state['timestamp'] = time.time()
        
        # Send to ESP32
        esp32_status = send_to_esp32(zoom_level, beam_angle)
        
        print(f" Zoom: {zoom_level:.2f}x |  Beam: {beam_angle:.1f}° | ESP32: {'yes' if esp32_status else 'no'}")
        
        return jsonify({
            'status': 'success',
            'zoom_received': zoom_level,
            'beam_angle': beam_angle,
            'esp32_sent': esp32_status
        }), 200
    
    return jsonify({'status': 'error', 'message': 'No zoom_level provided'}), 400

@app.route('/zoom', methods=['GET'])
def get_zoom():
    """Get current zoom and beamforming state"""
    return jsonify(current_state), 200

@app.route('/test_esp32', methods=['GET'])
def test_esp32():
    """Test ESP32 connection"""
    if esp32_serial is None:
        return jsonify({'status': 'disconnected'}), 503
    
    success = send_to_esp32(1.0, 0.0)
    return jsonify({'status': 'connected' if success else 'error'}), 200 if success else 500

@app.route('/')
def home():
    return f"""
    <h1>Audio-Visual Zoom System</h1>
    <p><strong>Current Zoom Level:</strong> {current_state['zoom_level']:.2f}x</p>
    <p><strong>Beam Angle:</strong> {current_state['beam_angle']:.1f}°</p>
    <p><strong>ESP32 Status:</strong> {'Connected' if esp32_serial else 'Disconnected'}</p>
    <p><strong>Last Updated:</strong> {time.ctime(current_state['timestamp'])}</p>
    <hr>
    <a href="/test_esp32">Test ESP32 Connection</a>
    """

if __name__ == '__main__':
    import socket
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)
    
    print(f"\n Audio-Visual Zoom System Starting...")
    print(f" Pi IP Address: {ip}")
    print(f" Server: http://{ip}:5000")
    print(f" Phone sends to: http://{ip}:5000/zoom")
    print(f" ESP32-S3: {'Connected' if esp32_serial else 'Not Connected'}\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)