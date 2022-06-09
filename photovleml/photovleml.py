from flask import Flask
from flask_cors import CORS

class PhotovleML(Flask):
    def __init__(self):
        Flask.__init__(self, __name__)
        CORS(self)

        # Set CORS middleware
        self.set_cors()
        
        # Init routers
        self.init_routers()
                
    def set_cors(self):
        pass
    
    def init_routers(self):
        @self.route("/")
        def index():
            return "Hello! PhotovleML"

        from .routers import ROUTERS

        for route in ROUTERS:
            self.register_blueprint(route)