from flask import Flask, request, jsonify
import time

app = Flask(__name__)

current_zoom = {
    'level': 1.0,
    'timestamp': time.time()
}

@app.route('/zoom', methods=['POST'])
def receive_zoom():
    """Receive zoom level from phone"""
    data = request.get_json()
    
    if 'zoom_level' in data:
        current_zoom['level'] = data['zoom_level']
        current_zoom['timestamp'] = time.time()
        
        print(f" Zoom Level: {data['zoom_level']:.2f}x")
        
        return jsonify({
            'status': 'success',
            'zoom_received': data['zoom_level']
        }), 200
    
    return jsonify({'status': 'error', 'message': 'No zoom_level provided'}), 400

@app.route('/zoom', methods=['GET'])
def get_zoom():
    """Get current zoom level"""
    return jsonify(current_zoom), 200

@app.route('/')
def home():
    return f"""
    <h1>Audio-Visual Zoom System</h1>
    <p>Current Zoom Level: {current_zoom['level']:.2f}x</p>
    <p>Last Updated: {time.ctime(current_zoom['timestamp'])}</p>
    """

if __name__ == '__main__':
    # Get Pi's IP address
    import socket
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)
    
    print(f"\n🚀 Zoom Server Starting...")
    print(f"📍 Pi IP Address: {ip}")
    print(f"🌐 Server running at: http://{ip}:5000")
    print(f"📱 Phone should send zoom data to: http://{ip}:5000/zoom\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)