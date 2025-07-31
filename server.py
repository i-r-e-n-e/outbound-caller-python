from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env.local")
print("transfer_to=", os.getenv("TRANSFER_TO"))

app = Flask(__name__)
CORS(app)

@app.route("/call-store", methods=["POST"])
def call_store():
    print("Incoming request")
    print("Headers:", dict(request.headers))
    print("JSON payload:", request.get_json())
    data = request.json
    phone_number = data.get("phoneNumber") # Getting the phonenumber from loveable

    # If no phone number is found get a 400 error
    if not phone_number:
        print("No phonenumber found in request body")
        return jsonify({"error": "Missing phone number"}), 400

    # Get the transfer number from the env
    transfer_to = os.getenv("TRANSFER_TO")
    if not transfer_to:
        print("transer_to not set")
        return jsonify({"error": "TRANSFER_TO not set in .env"}), 500

    # Prepare metadata
    metadata = {
        "phone_number": phone_number,
        "transfer_to": transfer_to 
    }

    # Convert metadata to a string
    metadata_str = str(metadata).replace("'", '"')

    # Run the LiveKit dispatch CLI command
    command = [
        "lk", "dispatch", "create",
        "--new-room",
        "--agent-name", "outbound-caller",
        "--metadata", metadata_str,
        "--api-key", os.environ["LIVEKIT_API_KEY"],
        "--api-secret", os.environ["LIVEKIT_API_SECRET"],
        "--url", os.environ["LIVEKIT_URL"]
    ]

    print("Running command", command)

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("subprocess output: ", result.stdout)
        return jsonify({"success": True})
    except subprocess.CalledProcessError as e:
        print("subprocess failed")
        print("return code: ", e.returncode)
        print("Stdout: ", e.stdout)
        print("stderr: ", e.stderr)
        return jsonify({"error": str(e)}), 500

# Run the program locally on port 3001
if __name__ == "__main__":
    app.run(port=3001)
