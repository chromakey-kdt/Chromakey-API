from fastapi import APIRouter


index_routers = APIRouter()

@index_routers.get("/")
def index():
    return "Hello! Workbench API Server"