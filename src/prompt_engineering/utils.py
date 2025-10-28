import base64
from pydantic import BaseModel
from enum import Enum
from openai import OpenAI
import os
from dotenv import load_dotenv
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


load_dotenv()

class Cats(str, Enum):
    socks = "Melvin"
    other = "other"

class WhichCat(BaseModel):
    cat: Cats
    confidence: float
    reasoning: str

class CatIdentifier():
    def __init__(self, model: str):
        self.model = model
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
    
    def identify(self, image_path: Path, system_prompt: str) -> str:
        response = self.client.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": system_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image(image_path)}"
                        }
                    }
                ]
                }
            ],
            response_format = WhichCat
        )
        return json.loads(response.choices[0].message.content)
    
    def identify_comp(self, image_path_sample: Path, image_path_test: Path, system_prompt: str) -> str:
        response = self.client.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": system_prompt},
                    {"type": "text", "text": "Sample image:"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image(image_path_sample)}"
                        }
                    },
                    {"type": "text", "text": "Test image:"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image(image_path_test)}"
                        }
                    }
                ]
                }
            ],
            response_format = WhichCat
        )
        return json.loads(response.choices[0].message.content)

def encode_image(image_path: str) -> str:
    """Encode image to base64 string for API request"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def compare_accuracy(preds: dict, label_df: pd.DataFrame):
    df_dict = dict(zip(label_df.image_id, label_df.label))
    matches = sum(1 for img_id in label_df.image_id if preds[img_id] == df_dict[img_id])
    return matches / len(label_df.image_id)

def evaluate_and_plot(preds: dict, label_df: pd.DataFrame, image_path: str, display_limit = 3):
    accuracy = compare_accuracy(preds, label_df)
    df_dict = dict(zip(label_df.image_id, label_df.label))
    
    wrong = [(img_id, preds[img_id]) for img_id in label_df.image_id if preds[img_id] != df_dict[img_id]]
    
    if len(wrong) == 0:
        print(f"Accuracy: {accuracy:.3f}, Great job!")
        return accuracy
    
    fig, axes = plt.subplots(1, min(len(wrong), display_limit), figsize=(12, 4))

    # Handle the case where axes might not be an array
    if not hasattr(axes, '__len__'):
        axes = [axes]
    
    for i, (img_id, pred_class) in enumerate(wrong[:display_limit]):
        img = plt.imread(f"{image_path}/{img_id}.jpg")
        axes[i].imshow(img)
        axes[i].set_title(f"Pred: {pred_class}. Label: {df_dict[img_id]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    print(f"Accuracy: {accuracy:.3f}")
    return accuracy
