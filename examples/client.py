from __future__ import annotations

import argparse
import json
import mimetypes
import uuid
from pathlib import Path
from urllib.request import Request, urlopen


def post_json(url: str, body: dict) -> dict:
    request = Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urlopen(request) as response:
        return json.load(response)


def upload(url: str, image: Path) -> dict:
    boundary = uuid.uuid4().hex
    content_type = mimetypes.guess_type(image.name)[0] or "application/octet-stream"
    body = (
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="cache_level"\r\n\r\n'
            f'visual_embedding\r\n--{boundary}\r\nContent-Disposition: form-data; name="image"; '
            f'filename="{image.name}"\r\nContent-Type: {content_type}\r\n\r\n'
        ).encode()
        + image.read_bytes()
        + f"\r\n--{boundary}--\r\n".encode()
    )
    request = Request(url, body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urlopen(request) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--server", default="http://localhost:8000")
    args = parser.parse_args()
    scene = upload(f"{args.server}/v1/scenes", args.image)
    print(json.dumps(scene, indent=2))
    for question in ("What objects are visible?", "Is the path blocked?"):
        result = post_json(
            f"{args.server}/v1/query",
            {
                "scene_id": scene["scene_id"],
                "question": question,
                "cache_level": "visual_embedding",
                "temperature": 0.0,
            },
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
