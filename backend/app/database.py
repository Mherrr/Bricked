import certifi
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFSBucket
from app.config import MONGO_URI, MONGO_DB

client: AsyncIOMotorClient = None
db = None
gridfs_bucket: AsyncIOMotorGridFSBucket = None


def _uses_tls(uri: str) -> bool:
    """Atlas (mongodb+srv://) and explicit tls/ssl=true URIs need TLS; a local mongod does not."""
    q = uri.lower()
    return q.startswith("mongodb+srv://") or "tls=true" in q or "ssl=true" in q


async def connect_db():
    global client, db, gridfs_bucket
    # Passing tlsCAFile turns TLS on, so only do it when the URI wants TLS —
    # otherwise a plain local MongoDB fails the handshake.
    kwargs = {"tlsCAFile": certifi.where()} if _uses_tls(MONGO_URI) else {}
    client = AsyncIOMotorClient(MONGO_URI, **kwargs)
    db = client[MONGO_DB]
    gridfs_bucket = AsyncIOMotorGridFSBucket(db, bucket_name="images")


async def close_db():
    if client:
        client.close()


def get_db():
    return db


def get_gridfs():
    return gridfs_bucket
