from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from emotion_classification import emotion_classifying
from predict_review_emotion import predict_emotion, emotion_feedback
from typing import List
import os

load_dotenv()
YOUR_IP = os.getenv("YOUR_IP")

app = FastAPI()

@app.get("/diary")
async def emotion(input_diary: str):
  try:
    # 감정 분류 함수 호출
    emo = ['공포', '놀람', '분노', '슬픔', '중립', '행복', '혐오']
    emotion_prediction = emotion_classifying(input_diary)
    feedback = emotion_feedback(input_diary)

    return JSONResponse(content={"emotion": emo[emotion_prediction], "wise": feedback})

  except RuntimeError as e:
    raise HTTPException(status_code=500, detail={f"RuntimeError: {e} \n It might be a problem with the prompt that user entered"})
  except Exception as e:
    raise HTTPException(status_code=500, detail={f"UnExpectedError: {e}"})

@app.get("/fortune")
async def predict_emotion(emotion: List[str]):
  try:
    prediction = predict_emotion(emotion)
    return JSONResponse(content={"prediction": prediction})
  
  except Exception as e:
    raise HTTPException(status_code=500, detail={f"UnExpectedError: {e}"})

if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app="main:app", host=YOUR_IP, port=9070, reload=True)