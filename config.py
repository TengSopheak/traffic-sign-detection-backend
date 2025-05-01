from dotenv import load_dotenv
import os

# Load the environment variables when config is imported
load_dotenv()

roboflow_api = os.getenv("roboflow_api")