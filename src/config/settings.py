from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_env: str = "dev"
    log_level: str = "INFO"

    openai_api_key: str = ""

    vector_db_dir: str = "./local_chroma"
    datasets_dir: str = "./datasets"
    reports_dir: str = "./reports"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

settings = Settings()