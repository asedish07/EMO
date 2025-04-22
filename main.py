# from fastapi import FastAPI, HTTPException
# from fastapi.responses import JSONResponse, FileResponse
# from fastapi.middleware.cors import CORSMiddleware
# from key import YOUR_IP

# app = FastAPI()

# @app.get("/emotion")
# async def emotion(input_diary: str):
#   try:
#     # 감정 분류 함수 호출
    

#     # 피드백 함수 호출
    
#   except Exception as e:
#     raise HTTPException(status_code=500, detail={f"UnExpectedError: {e}"})

# if __name__ == "__main__":
#   import uvicorn
#   uvicorn.run(app="main:app", host=YOUR_IP, port=5070, reload=True)