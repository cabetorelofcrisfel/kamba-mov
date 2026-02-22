from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()

    with open("frame.jpg", "wb") as f:
        f.write(contents)

    return {"status": "ok"}

@app.get("/")
def root():
    return {"server": "Kamba Mov API Online"}