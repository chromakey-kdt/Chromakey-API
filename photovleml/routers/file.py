import os
from fastapi import APIRouter, File, UploadFile


file_routers = APIRouter()

@file_routers.post("/")
async def upload_files(file: UploadFile = File(...)):
    UPLOAD_DIRECTORY = "./"

    contents = await file.read()
    with open(os.path.join(UPLOAD_DIRECTORY, file.filename), "wb") as fp:
        fp.write(contents)

    print(file.filename)
    # return {"filenames": [file.filename for file in files]}
    return "Success"