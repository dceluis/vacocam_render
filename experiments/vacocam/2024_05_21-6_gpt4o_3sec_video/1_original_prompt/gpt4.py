from openai import OpenAI

import json
import os

from typing import Union

def submit_image(encoded_image: Union[str, list[str]], encoded_metadata: str, version: str = "v1"):
    api_key = os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    prompts = {
    "v3": f"""
Look at these stills from 3 seconds of football video. 
The colored area(s) with a letter label are clusters of ball detections across multiple frames.

The areas metadata is the following:

{encoded_metadata}

Your task is to determine which areas are ball detections belonging to the primary match, using their positioning, context, trajectory, time, etc.
There may or not be more than one match being played. Also, an object that is not a ball may have been labeled as a ball.

Pay close attention to the lines, goals and other markers to determine the bounds and action of the primary match.
The drawn arrows indicate the general direction of the detections.

You only need your visual understanding of the image as-is to accomplish this task, you dont need any additional information.

Respond with a top-level array of objects with keys "id": String, "primary": Boolean, "reason" String ["primary ball", "irrelevant ball", "not ball"].
Respond with valid JSON stripped of any display formatting (including blockquotes, etc.), comments, or explanations
Respond with labels ordered from most to least relevant to the primary match.
notalk; justgo
    """
    }

    prompt = prompts[version]
    if prompt is None:
        print("[gpt4o.py] ERROR: invalid prompt version. Using v3")
        prompt = prompts["v3"]

    prompt_parts = []

    if isinstance(encoded_image, str):
        prompt_parts.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{encoded_image}",
                "detail": "low",
            }
        })
    else:
        for image_data in encoded_image:
            image_part = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_data}",
                    "detail": "low",
                }
            }
            prompt_parts.append(image_part)

    prompt_parts.append(prompt)

    messages=[
        {
            "role": "user",
            "content": [
                *prompt_parts,
                {
                    "type": "text",
                    "text": prompt,
                },
            ]
        }
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=500,
            temperature=0.1,
        )

        response_dump = response.model_dump_json()

        tmp_file = "/home/luis/workspace/vacocam_render/experiments/vacocam/gpt4.txt"

        with open(tmp_file, "w") as f:
            f.write("#########" * 10)
            f.write("messages:")
            f.write(str(messages))
            f.write("#########" * 10)
            f.write("\n")
            f.write("#########" * 10)
            f.write("response:")
            f.write(str(response_dump))
            f.write("#########" * 10)
            f.write("\n")

        return json.loads(response_dump)
    except Exception as e:
        # call the police
        print("ERROR")
        print(e)
        return None