from anthropic import Anthropic

import json
import os

def submit_image(encoded_image, encoded_metadata, version="v1"):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    client = Anthropic(api_key=api_key)

    prompts = {
        "v1": f"""
Look at this still from 20 seconds of football video. 
The colored area(s) with a letter label are clusters of ball positions already detected.

The areas metadata is the following:

{encoded_metadata}

Your task is to determine which areas (if any) do not belong to the primary match, using their positioning, context, trajectory, etc.
There may or not be more than one match being played.

Pay close attention to the lines, goals and other markers to determine the bounds of the primary match.
You only need your visual understanding of the image as-is to accomplish this task, you dont need any additional information.

Respond with a top-level array of objects with keys "id": String and "primary": Boolean.
Respond with valid JSON stripped of any display formatting (including blockquotes, etc.), comments, or explanations
Respond with labels ordered from most to least relevant to the primary match.
notalk; justgo
    """,
    "v2": f"""
Look at this still from 10 seconds of football video. 
The colored area(s) with a letter label are clusters of ball detections.

The areas metadata is the following:

{encoded_metadata}

Your task is to determine which areas are ball detections belonging to the primary match, using their positioning, context, trajectory, time, etc.

There may or not be more than one match being played. Also, an object that is not a ball may have been labeled as a ball.

Pay close attention to the lines, goals and other markers to determine the bounds and action of the primary match.
Big time overlaps and far distances usually mean different matches.
Small time overlaps and close proximity usually mean areas belong to the same match.

You only need your visual understanding of the image as-is to accomplish this task, you dont need any additional information.

Respond with a top-level array of objects with keys "id": String, "primary": Boolean, "reason" String ["primary ball", "irrelevant ball", "not ball"].
Respond with valid JSON stripped of any display formatting (including blockquotes, etc.), comments, or explanations
Respond with labels ordered from most to least relevant to the primary match.
notalk; justgo
    """,
    "v3": f"""
Look at this still from 5 seconds of football video. 
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
        print("[claude-3.py] ERROR: invalid prompt version. Using v3")
        prompt = prompts["v3"]

    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": encoded_image
                    }
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ]
        }
    ]

    try:
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            messages=messages,
            max_tokens=500,
            temperature=0.1,
        )

        response_dump = response.model_dump_json()

        tmp_file = os.path.join(os.path.dirname(__file__), "claude-3.txt")

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