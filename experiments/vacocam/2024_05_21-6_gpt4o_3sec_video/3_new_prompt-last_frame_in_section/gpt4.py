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
    "v4": """
Look at these stills from 3 consecutive seconds of football video.
The colored areas with a letter label are clusters of ball detections across multiple frames.
The drawn arrows indicate the general direction of the detection areas.

The areas metadata is:

{encoded_metadata}

You are an expert at determining which areas are ball detections belonging to the primary match, using their positioning, context, trajectory, players' movement and proximity, etc.
Also, you accurately identify objects incorrectly detected and labeled as balls.

More than one match may be visible, but you focus on the primary one.
You pay close attention to the lines, goals, and other objects to determine the bounds of the primary match.

Your task is the following:

Describe in one sentence what is happening in the 3-second sequence of the match.
Focus on highlighting the key action observed from the positioning and movement of the primary ball.
Your description is part of a sequence for providing hints to a system without visual access to the footage.
Get to the point describing the observed action. Be authoritative, brief and confident.

Then, categorize the colored areas into "primary ball", "irrelevant ball" and "not ball".

Finally identify the area or areas that represent the primary ball.

You only need your visual understanding of the images to accomplish this task, you don't need any additional information.

Respond in valid JSON with schema:
{{
    "sequence_description": String,
    "areas": List[
        {{
            "id": String,
            "category": String["primary ball" | "irrelevant ball" | "not ball"],
            "primary": Boolean
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