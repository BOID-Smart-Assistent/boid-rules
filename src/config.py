import os
import json

from dotenv import load_dotenv
from llama_index.legacy.llms import Ollama
from model.boid import Schedule, User
from ollama import Client

load_dotenv()

class Config:
    online_mode = os.getenv('ONLINE_MODE', 'False').lower() in ('true', '1', 't')
    websocket_url = os.getenv('WEBSOCKET_URL')
    debug = os.getenv('DEBUG_MODE', 'False').lower() in ('true', '1', 't')
    llm = Ollama(model=os.getenv('LLM_MODEL'), base_url=os.getenv('LLM_HOST'), request_timeout=1600)

    schedule: Schedule
    user: User

    def __init__(self):
        if self.online_mode:
            pass
        else:
            with open('./data/schedule.json', 'r') as file:
                self.schedule = Schedule().from_json(file.read())
            with open('./data/user.json', 'r') as file:
                self.user = User().from_json(file.read())

    def set_schedule(self, schedule: Schedule):
        self.schedule = schedule

    def set_user(self, user: User):
        self.user = user

config = Config()
