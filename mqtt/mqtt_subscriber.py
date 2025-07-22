import time
from paho.mqtt import client as mqtt_client

# Configuration
broker = "192.168.0.36"
port = 1883
topic = "smartwound/gas"  # wildcard to receive all vitals

# Callback when a message is received
def on_message(client, userdata, msg):
    try:
        payload = msg.payload.decode("utf-8", errors="replace")
        print(f"📥 Received on `{msg.topic}`: {payload}")
    except Exception as e:
        print(f"⚠️ Error decoding message: {e}")

# Callback when connected to broker
def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print("✅ Connected to MQTT Broker!")
        client.subscribe(topic)
        print(f"🔔 Subscribed to `{topic}`")
    else:
        print(f"❌ Connection failed with code {reason_code}")

# Callback for disconnection
def on_disconnect(client, userdata, reason_code):
    print(f"🔌 Disconnected (reason_code={reason_code})")
    if reason_code != 0:
        print("⚠️ Unexpected disconnection. Attempting to reconnect...")
        reconnect(client)

# Reconnect logic
def reconnect(client):
    while True:
        try:
            client.username_pw_set("ujwal__gp", "root")
            client.reconnect()
            print("🔁 Reconnected to broker.")
            return
        except:
            print("⏳ Reconnect failed. Retrying in 3s...")
            time.sleep(3)

# MQTT client setup
def main():
    client = mqtt_client.Client(
        callback_api_version=mqtt_client.CallbackAPIVersion.VERSION2
    )
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message

    try:
        client.username_pw_set("ujwal__gp", "root")
        client.connect(broker, port)
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return

    client.loop_start()

    try:
        print("📡 Listening for messages (press Ctrl+C to stop)...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user.")
    finally:
        client.loop_stop()
        client.disconnect()
        print("👋 Disconnected cleanly.")

if __name__ == "__main__":
    main()
