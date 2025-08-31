from datetime import datetime
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import ConnectionFailure
from typing import Optional
from pydantic import BaseModel, Field
import os

# --- 1. Data Model for an Image Processing Log ---
class ImageLog(BaseModel):
    original_image_url: str
    processed_image_url: str
    filter_applied: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        from_attributes = True

# --- 2. Database Connection Class ---
class Database:
    """Handles MongoDB connection and provides collection access."""
    def __init__(self, uri: Optional[str] = None, db_name: Optional[str] = None):
        self.client = None
        self.db = None
        self.logs_collection = None
        
        # Only connect if MongoDB URI is provided
        if uri:
            try:
                self.client = MongoClient(uri, tls=True, tlsAllowInvalidCertificates=True)
                self.client.admin.command('ismaster') # Check for a successful connection
                print("✅ MongoDB connection successful.")
                self.db = self.client[db_name or "pixel_codex_db"]
                self.logs_collection: Collection = self.db['image_processing_logs']
            except ConnectionFailure as e:
                print(f"❌ Could not connect to MongoDB: {e}")
                self.client = None
        else:
            print("ℹ️ MongoDB URI not provided, running without database.")
    
    def close(self):
        """Closes the database connection."""
        if self.client:
            self.client.close()

# --- 3. CRUD Operation ---
def create_log_entry(collection: Optional[Collection], log_data: ImageLog) -> Optional[str]:
    """Inserts a new image processing log into the collection."""
    if not collection:
        print("⚠️ Database collection not available, skipping log entry.")
        return None
        
    try:
        result = collection.insert_one(log_data.model_dump())
        print(f"📄 Logged operation to MongoDB with ID: {result.inserted_id}")
        return str(result.inserted_id)
    except Exception as e:
        print(f"❌ Error logging to MongoDB: {e}")
        return None