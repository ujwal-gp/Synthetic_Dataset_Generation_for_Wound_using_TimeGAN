import subprocess
import datetime

# Configuration
interface = "ens160"  # Use "lo" for localhost, "eth0" or "wlan0" for real interfaces
duration = 30     # Seconds to capture
publisher_ip = "192.168.0.36"
output_file = f"mqtt_capture_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pcapng"
csv_file = f"mqtt_capture_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
display_filter = "mqtt"  # Capture only MQTT packets

# Build tshark command
packet_capture_command = [
    "tshark",
    "-i", interface,
    "-a", f"duration:{duration}",
    "-f", f"host {publisher_ip} and tcp port 1883",  # BPF filter for MQTT
    "-w", output_file
]

csv_conversion_command = [
    "tshark",
    "-r", output_file,
    "-Y", "mqtt",
    "-T", "fields",
    "-e", "frame.time",
    "-e", "ip.src",
    "-e", "ip.dst",
    "-e", "tcp.srcport",
    "-e", "tcp.dstport",
    "-e", "mqtt.msgtype",
    "-e", "mqtt.topic",
    "-e", "mqtt.msg",
    "-E", "header=y",
    "-E", "separator=,",
    "-E", "quote=d",
    "-E", "occurrence=f"
]

print(f"🚀 Capturing packets on interface '{interface}' for {duration} seconds...")
print(f"📁 Output: {output_file}")

try:
    subprocess.run(packet_capture_command, check=True)
    print("✅ Capture complete.")
    with open(csv_file, "w") as outfile:
        subprocess.run(csv_conversion_command, stdout=outfile, check=True)
        print("✅ CSV conversion complete.")

except subprocess.CalledProcessError as e:
    print(f"❌ Error during capture: {e}")
