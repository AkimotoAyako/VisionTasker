import os
import requests
import base64
from mimetypes import guess_type

# Function to encode a local image into data URL 
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"

# Configuration
IMAGE_PATH = f"data\screenshot\screenshot_gpt4.png"
# IMAGE_PATH = r'G:\GUI_detection_grouping\task1\wechat_me_service.jpg'
GPT4V_KEY = "75e26ccc637440f19f1688a8fbf111a4"
encoded_image = base64.b64encode(open(IMAGE_PATH, 'rb').read()).decode('ascii')
# encoded_image = local_image_to_data_url(IMAGE_PATH)

headers = {
    "Content-Type": "application/json",
    "api-key": GPT4V_KEY,
}

# Payload for the request
payload = {
  "messages": [
    {
      "role": "system",
      "content": [
          {
              "type": "text",
              "text": "You are an AI assistant that helps people find information."
          }
      ]
    },
    {
      "role": "user",
      "content": [
          {
              "type": "image_url",
              "image_url":
                  {
                      "url": f"data:image/png;base64,{encoded_image}"
                  }
          },
          {
              "type": "text",
              "text": "Describe this image in a single sentence."
          }
      ]
    }
  ],
  "temperature": 0.7,
  "top_p": 0.95,
  "max_tokens": 800
}

# GPT4V_ENDPOINT = "https://gpt-turbo-4.openai.azure.com/openai/deployments/gpt-4-vision-preview/chat/completions?api-version=2023-07-01-preview"
# GPT4V_ENDPOINT = "https://gpt-4-v.openai.azure.com/openai/deployments/gpt-4-1106/chat/completions?api-version=2024-02-15-preview"
GPT4V_ENDPOINT = "https://gpt-turbo-4.openai.azure.com/openai/deployments/gpt-4-vision/chat/completions?api-version=2024-02-15-preview"


# print(payload)
# Send request
try:
    response = requests.post(GPT4V_ENDPOINT, headers=headers, json=payload)
    response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
except requests.RequestException as e:
    raise SystemExit(f"Failed to make the request. Error: {e}")

# Handle the response as needed (e.g., print or process)
data = response.json()
# print(data)
print(data)
total_tokens = data['usage']['total_tokens']
print(f"total_tokens: {total_tokens}\n")

result = data['choices'][0]['message']['content']
print(f"GPTV4 answer: {result}\n")
