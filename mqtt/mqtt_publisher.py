import pandas as pd
import time
import json
import random
import string
from paho.mqtt import client as mqtt_client

# MQTT Broker Configuration
broker = 'localhost'
port = 1883
topic = "smartwound/gas"  # Make sure topic is valid
NUM_CLIENTS = random.randint(5, 15)

# 🔐 Safe topic check
def is_valid_topic(topic):
    if not topic or " " in topic:
        return False
    try:
        topic.encode("utf-8")
        return True
    except UnicodeEncodeError:
        return False

# Generate unique client ID with 12–20 characters
def generate_random_client_id():
    length = random.randint(12, 25)
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

# Connect one MQTT client
def connect_mqtt(client_id):
    client = mqtt_client.Client(client_id=client_id)
    try:
        client.connect(broker, port)
        print(f"✅ Connected client ID: {client_id}")
        return client
    except Exception as e:
        print(f"❌ Failed to connect {client_id}: {e}")
        return None

# Publish data row-by-row from CSV
def publish_data(client, df, delay=0.05):
    client.loop_start()
    for _, row in df.iterrows():
        payload = row.to_dict()

        if not is_valid_topic(topic):
            print(f"⚠️ Skipping invalid topic: {topic}")
            continue

        result = client.publish(topic, json.dumps(payload))
        status = result[0]
        if status == 0:
            print(f"📤 {client._client_id.decode()} ➜ `{topic}`: {payload}")
        else:
            print(f"❌ Failed to send message for {client._client_id.decode()}")
        time.sleep(delay)
    client.loop_stop()

# Main routine
if __name__ == "__main__":
    print(f"🚀 Simulating {NUM_CLIENTS} clients...\n")
    df = pd.read_csv("../synthetic_wound_gas_flat.csv")
    clients = []

    for _ in range(NUM_CLIENTS):
        cid = generate_random_client_id()
        client = connect_mqtt(cid)
        if client:
            clients.append(client)

    if not clients:
        print("❌ No clients could connect. Exiting.")
        exit(1)

    try:
        for client in clients:
            publish_data(client, df, delay=random.uniform(0.05, 0.2))
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
    finally:
        for c in clients:
            c.disconnect()
