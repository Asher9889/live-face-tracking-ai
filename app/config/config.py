import os
from dotenv import load_dotenv

load_dotenv()

FRAME_RATE = os.getenv("FRAME_RATE", "15")