from datetime import datetime
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import ConnectionFailure
from typing import Optional
from pydantic import BaseModel, Field

# --- 1. Data Model for an Image Processing Log ---
class ImageLog(BaseModel):
    """Defines the structure of a log entry in MongoDB."""
    original_filename: str
    processed_filename: str
    filter_applied: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        from_attributes = True


# --- 2. Database Connection Class ---
class Database:
    """Handles MongoDB connection and provides collection access."""
    def __init__(self, uri: str, db_name: str):
        try:
            self.client = MongoClient(uri)
            self.client.admin.command('ismaster') # Check for a successful connection
            print("✅ MongoDB connection successful.")
        except ConnectionFailure as e:
            print(f"❌ Could not connect to MongoDB: {e}")
            self.client = None
        
        if self.client:
            self.db = self.client[db_name]
            self.logs_collection: Collection = self.db['image_processing_logs']

    def close(self):
        """Closes the database connection."""
        if self.client:
            self.client.close()


# --- 3. CRUD Operation ---
def create_log_entry(collection: Collection, log_data: ImageLog) -> str:
    """Inserts a new image processing log into the collection."""
    try:
        result = collection.insert_one(log_data.model_dump())
        print(f"📄 Logged operation to MongoDB with ID: {result.inserted_id}")
        return str(result.inserted_id)
    except Exception as e:
        print(f"❌ Error logging to MongoDB: {e}")
        return None