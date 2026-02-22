import os


class Settings:
    PROJECT_NAME: str = "Titanic Survival API"
    VERSION: str = "1.0.0"
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/model.pkl")


settings = Settings()
