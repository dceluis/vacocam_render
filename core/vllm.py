from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

import json
import os

from typing import Union

def submit_image(encoded_image: Union[str, list[str]], encoded_metadata: str, vllm="claude3"):
    prompt = f"""
Look at these images from 3 seconds of football video. 
Each image is the last frame of a 1-second video clip.
The colored area(s) with a letter label are balls detected by an AI model in the preceding frames of each 1-second clip.
They have been added to the last frame for context.
The AI model is not perfect and may have detected non-ball objects as balls.

The areas metadata is:
{encoded_metadata}

Your task is to determine which areas, if any, are true ball detections belonging to the primary match that is the focus of the video.

Pay close attention to the lines, goals and other markers to determine the bounds and actions of the primary match.
The drawn arrows indicate the general direction of the detections.

Respond with a top-level array of objects with keys "id": String, "primary": Boolean, "reason" String ["primary ball", "irrelevant ball", "not ball"].
Respond with valid JSON stripped of any display formatting (including blockquotes, etc.), comments, or explanations
Respond with areas ordered by relevancy to the primary match, descending.
notalk; justgo
    """

# You only need your visual understanding of the image as-is to accomplish this task, you dont need any additional information.

    if vllm == "gpt4":
        api_key = os.environ.get("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key)

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

    elif vllm == "claude3":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        client = Anthropic(api_key=api_key)

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
            return json.loads(response_dump)
        except Exception as e:
            # call the police
            print("ERROR")
            print(e)
            return None
    elif vllm == "gemini1.5":
        api_key = os.getenv("GOOGLE_API_KEY")

        generation_config = {
            "temperature": 0.0,
            "top_p": 1,
            "top_k": 32,
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

        try:
            response = model.generate_content(prompt_parts)

            response_dump = response.text

            tmp_file = os.path.join(os.path.dirname(__file__), "gemini-1.5-flash.txt")

            with open(tmp_file, "w") as f:
                f.write("#########" * 10)
                f.write("messages:")
                f.write(str(prompt_parts))
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
    else:
        raise ValueError("Invalid VLLM model")