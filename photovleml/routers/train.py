from flask import Blueprint, jsonify
from photovleml.service import PhotovleService


train_bp = Blueprint('train', __name__, url_prefix='/train')

@train_bp.route("/")
def train_model():
    PhotovleService.train()
    
    return jsonify(True)
    # return "hello"