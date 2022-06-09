import os
from .train import trainer
from .predict import predictor, video_predictor


class PhotovleService:
    @staticmethod
    def train(user_id, timestamp):
        trainer(user_id, timestamp)
    
    @staticmethod
    def predict(user_id, timestamp):
        return predictor(user_id, timestamp)

    @staticmethod
    def predict_video(user_id, timestamp):
        return video_predictor(user_id, timestamp)
