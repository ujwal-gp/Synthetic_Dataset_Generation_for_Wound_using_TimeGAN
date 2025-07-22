# server.py
import flwr as fl

def main():
    # Configure and start Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3)
    )

if __name__ == "__main__":
    main()
