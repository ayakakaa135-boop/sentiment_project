import os

bind = f"0.0.0.0:{os.getenv('PORT', '10000')}"
timeout = 600

# Keep single worker by default for free-tier memory limits; allow override.
workers = int(os.getenv("WEB_CONCURRENCY", "1"))
