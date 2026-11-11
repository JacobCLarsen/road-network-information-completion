from pydantic_settings import BaseSettings, SettingsConfigDict

# The newest migration from the data repo (astep-aau/data) that our models are written for.
# If the database has not run this migration yet, the API will not start.
# Remember to update this when new migrations are added in the data repo.
EXPECTED_MIGRATION = "attribute_prediction/0018_metric.sql"


class Settings(BaseSettings):
    # Looks for environment variables first, and if they are not set it reads the .env file.
    # Locally we use .env, on the server the values come from the kubernetes secret.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Example: postgresql+asyncpg://postgres:postgres@localhost:5432/data
    database_url: str
    # If true, all SQL queries are printed (nice for debugging)
    debug: bool = False
    # Set to false to skip the migration check when the API starts
    db_check_migrations: bool = True


settings = Settings()
