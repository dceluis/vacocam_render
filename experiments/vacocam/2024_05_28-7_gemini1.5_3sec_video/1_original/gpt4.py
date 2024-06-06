from pathlib import Path
import google.generativeai as genai

import PIL.Image

import json
import os

from typing import Union

api_key = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

def submit_image(encoded_image: Union[str, list[str]], encoded_metadata: str, version: str = "v1"):

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
        print("[gemini-1_0.py] ERROR: invalid prompt version. Using v3")
        prompt = prompts["v3"]
    
    generation_config = {
        "temperature": 0.0,
        "top_p": 1,
        "top_k": 8,
        "max_output_tokens": 4096,
    }

    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
    ]

    model = genai.GenerativeModel(model_name="gemini-1.5-flash",
                                  generation_config=generation_config,
                                  safety_settings=safety_settings)

    prompt_parts = []

    if isinstance(encoded_image, str):
        prompt_parts.append({
            "mime_type": "image/png",
            "data": encoded_image
        })
    else:
        for image_data in encoded_image:
            image_part = {
                "mime_type": "image/png",
                "data": image_data
            }
            prompt_parts.append(image_part)

    prompt_parts.append(prompt)

    # try:
    response = model.generate_content(prompt_parts)

    response_dump = response.text

    tmp_file = os.path.join(os.path.dirname(__file__), "gemini-3.txt")

    with open(tmp_file, "w") as f:
        f.write("#########" * 10)
        f.write("\n")
        f.write("messages:")
        f.write("\n")
        f.write(str(prompt_parts))
        f.write("#########" * 10)
        f.write("\n")
        f.write("\n")
        f.write("#########" * 10)
        f.write("\n")
        f.write("response:")
        f.write("\n")
        f.write(str(response_dump))
        f.write("#########" * 10)
        f.write("\n")

    return json.loads(response_dump)
    # except Exception as e:
    #     # call the police
    #     print("ERROR")
    #     print(e)
    #     return None