
import asyncio
import json
import argparse
import sys
import time
import websockets


async def listen(uri: str, timeout: float, max_messages: int):
    try:
        async with websockets.connect(uri, open_timeout=10) as ws:
            received = 0
            deadline = time.monotonic() + timeout
 
            while received < max_messages:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    print(f"\nTimeout reached after {timeout}s with {received} message(s) received.")
                    break
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                except asyncio.TimeoutError:
                    print(f"\nNo message received within {timeout}s.")
                    break
 
                received += 1
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError as e:
                    print(f"[#{received}] INVALID JSON: {e}")
                    print(f"  Raw: {raw[:200]}")
                    continue
 
                # validate
                ok = True

                for field in ("camera", "timestamp", "matches"):
                    if field not in payload:
                        print(f"[#{received}] MISSING field '{field}' in payload")
                        ok = False
 
                if ok:
                    n_matches = len(payload.get("matches", []))
                    print(f"[#{received}] Valid payload")
                    print(f"  camera    : {payload['camera']}")
                    print(f"  timestamp : {payload['timestamp']}")
                    print(f"  matches   : {n_matches}")

                    for m in payload["matches"]:
                        print(f"    uuid={m.get('uuid')}  "
                              f"sim={m.get('similarity'):.4f}  "
                              f"det_conf={m.get('det_conf'):.4f}  "
                              f"bbox={m.get('bbox')}")
                    print()
 
            if received == 0:
                print("No messages received")
                sys.exit(1)
            else:
                print(f"Done. Received {received} message(s).")
 
    except (ConnectionRefusedError, OSError) as e:
        print(f"Could not connect: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test hpc_server WebSocket feed")

    parser.add_argument("--uri",          default="ws://localhost:8765",
                        help="WebSocket URI to connect to (default: ws://localhost:8765)")
    parser.add_argument("--timeout",      type=float, default=60.0,
                        help="Seconds to wait for the first message (default: 60)")
    parser.add_argument("--max-messages", type=int,   default=5,
                        help="Stop after this many messages (default: 5)")
    args = parser.parse_args()
    
    asyncio.run(listen(args.uri, args.timeout, args.max_messages))
