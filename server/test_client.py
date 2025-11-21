import io
import base64
import json
import requests
from pathlib import Path

from scipy.io import wavfile

GENDER: str = "female"
AGE: int = 37
# HOST: str = "http://localhost:7310"
HOST: str =  "http://35.238.85.234:7311"
PATH: str = "/qc-edd/parkinson"
# WAV_PATH: Path = Path(__file__).with_name("audio_input_ui.wav")
WAV_PATH: Path = Path(__file__).with_name("test_audio.wav")

def main() -> None:
    #with open(WAV_PATH, "rb") as f:
    #    sr, audio_array = wavfile.read(f)
    # Send the PCM array instead of the WAV bytes:
    #audio_b64 = base64.b64encode(audio_array.tobytes()).decode("ascii")    
    
    with open(WAV_PATH, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("ascii")
    payload: dict = {
        "audio": audio_b64,
        "gender": GENDER,
        "age": AGE,
    }
    headers = {"Content-Type": "application/json"}
    print(f"Sending POST to: {HOST}{PATH}")
    print(f"Headers: {json.dumps(headers, indent=2)}")
    preview = dict(payload)
    if isinstance(preview.get("audio"), str) and len(preview["audio"]) > 120:
        preview["audio"] = preview["audio"][:120] + "...(truncated)"
    print(f"Payload: {json.dumps(preview, indent=2)}")
    for _ in range(1): # 1_000_000
        try:
            resp = requests.post(f"{HOST}{PATH}", headers=headers, json=payload) # timeout=???
            status = resp.status_code
            resp.raise_for_status()
            print("OK")
            print(f"Status Code: {status}")
            try:
                print(json.dumps(resp.json(), indent=2))
            except ValueError:
                print(resp.text)
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error: {e}")
            print(f"Status Code: {getattr(e.response, 'status_code', 'unknown')}")
            print(f"Body: {getattr(e.response, 'text', '')}")
        except requests.exceptions.Timeout as e:
            print(f"Timeout: {e}")
        except requests.exceptions.ConnectionError as e:
            print(f"Connection error: {e}")
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")

if __name__ == "__main__":
    main()
