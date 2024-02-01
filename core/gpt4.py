from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

import json
import os

def submit_image(encoded_image, encoded_metadata):
    api_key = os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    prompt = f"""
Look at this still from 10 seconds of football video. 
The colored area(s) with a letter label are clusters of ball detections across multiple frames.

The areas metadata is:
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

    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encoded_image}",
                        "detail": "low",
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
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=messages,
            max_tokens=500,
            temperature=0.1,
        )

        response_dump = response.model_dump_json()
        return json.loads(response_dump)
    except Exception as e:
        # call the police
        print("ERROR")
        print(e)
        return None