from dotenv import load_dotenv
from google import genai
from google.genai import types
from typing import List
import os

load_dotenv()

API_KEY = os.getenv("API_KEY")


def predict_emo(emotions: List[str]):
  emotion = ""
  for i in range(len(emotions)):
    if i == (len(emotions) - 1):
      emotion += str(emotions[i])
    else:
      emotion += str(emotions[i]) + ", "
    
def predict_emotion(emotion: List[str]):
  client = genai.Client(api_key=API_KEY)
  response = client.models.generate_content(
      model="gemini-2.0-flash",
      config=types.GenerateContentConfig(
          system_instruction="내 이전 날들의 감정들을 보내줄게. 내가 지금까지 하루하루 느낀 감정들을 보고 내가 오늘 느끼는 감정을 예측해줘. 짧은 한 문장으로 설명해줘."),
      contents=emotion
  )

  return response.text

def emotion_feedback(input_diary: str):
  client = genai.Client(api_key=API_KEY)
  response = client.models.generate_content(
      model="gemini-2.0-flash",
      config=types.GenerateContentConfig(
          system_instruction="너는 심리 상담가야. 지금 너에게 나의 오늘에 대한 감정일기를 보내줄게. 응원의 말을 담아 오늘의 나에 대해 짧게 리뷰해줘."),
      contents=input_diary
  )

  return response.text