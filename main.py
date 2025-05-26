from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from emotion_classification import emotion_classifying
from predict_review_emotion import predict_emo, emotion_feedback
from typing import List
from pydantic import BaseModel
import os

load_dotenv()
YOUR_IP = os.getenv("YOUR_IP")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 출처에서 접근 허용 (필요에 맞게 수정 가능)
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메소드 허용 (POST, GET 등)
    allow_headers=["*"],  # 모든 헤더 허용
)

# request_body
class Diary(BaseModel):
  input_diary: str

class Emotion(BaseModel):
  emotion: List[str]

@app.post("/diary")
def emotion(input_diary: Diary):
  try:
    # 감정 분류 함수 호출
    emo = ['공포', '놀람', '분노', '슬픔', '중립', '행복', '혐오']
    emotion_prediction = emotion_classifying(input_diary.input_diary)
    if not 0 <= emotion_prediction < len(emo):
      raise ValueError(f"Invalid emotion index: {emotion_prediction}")
    feedback = emotion_feedback(input_diary.input_diary)

    return JSONResponse(content={"emotion": emo[emotion_prediction], "wise": feedback})

  except RuntimeError as e:
    raise HTTPException(status_code=500, detail={f"RuntimeError: {str(e)} \n It might be a problem with the prompt that user entered"})
  except Exception as e:
    raise HTTPException(status_code=500, detail={f"UnExpectedError: {str(e)}"})

@app.post("/fortune")
async def predict_emotion(emotion: Emotion):
  try:
    prediction = predict_emo(emotion.emotion)
    return JSONResponse(content={"prediction": prediction})
  
  except Exception as e:
    raise HTTPException(status_code=500, detail={f"UnExpectedError: {str(e)}"})

if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app="main:app", host=YOUR_IP, port=6600, reload=True)