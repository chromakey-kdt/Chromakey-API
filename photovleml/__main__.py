import os
import torch
import numpy as np
import random
from dotenv import load_dotenv
from .photovleml import PhotovleML

def fix_seeds(random_seed=42):
    torch.manual_seed(random_seed)

    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    np.random.seed(random_seed)
    random.seed(random_seed)
    
if __name__ == "__main__":
    fix_seeds()
    
    # Load .env file
    load_dotenv(verbose=True)
    
    # Run PhotovleML API server    
    if os.path.isdir(os.path.join(os.getenv("TEMP_DATA_PATH"))):
        import shutil

        shutil.rmtree(os.path.join(os.getenv("TEMP_DATA_PATH")))

    app = PhotovleML()
    app.run(
        host=os.getenv("SERVER_HOST"),
        port=int(os.getenv("SERVER_PORT")),
        debug=True
    )

    # For debugging
    # from .service.photovle_service import PhotovleService
    #
    # PhotovleService.predict("136954524")