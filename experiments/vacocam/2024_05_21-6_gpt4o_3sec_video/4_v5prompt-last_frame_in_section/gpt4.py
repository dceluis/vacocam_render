from openai import OpenAI

import json
import os

from typing import Union

def submit_image(encoded_image: Union[str, list[str]], encoded_metadata: str, version: str = "v1"):
    api_key = os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    prompts = {
    "v3": """
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
    """.format(encoded_metadata=encoded_metadata),
    "v5": """
Look at these stills from 3 consecutive seconds of football video.
The colored areas with a letter label are clusters of ball detections across multiple frames.
The drawn arrows indicate the general direction of the detection areas.

The areas metadata is:

{encoded_metadata}

You are an expert at determining which areas are the most interesting to the primary match, using their positioning, context, trajectory, players' movement and proximity, etc.

More than one match may be visible, but you focus on the primary one.
You pay close attention to the lines, goals, and other objects to determine the bounds of the primary match.

Your task is the following:

Describe the approximate distance where the main action of the game is occurring.

Finally, order by importance, descending, and pick the most important area to focus on, that is, the one that follows and captures the most information from the match being recorded.

You only need your visual understanding of the images to accomplish this task, you don't need any additional information.

Respond in valid JSON stripped of any display formatting (including blockquotes, etc.), comments, or explanations, with schema:
{{
    "distance": String["near"|"far"|"mid"],
    "areas": List[
        {{
            "id": String,
            "focus": Boolean
        }}
    ],
}}
    """.format(encoded_metadata=encoded_metadata)
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
            temperature=0.0,
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