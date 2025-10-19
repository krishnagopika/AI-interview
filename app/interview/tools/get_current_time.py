# utils.py
from datetime import datetime

def get_current_time():
    """Return current date in MM-DD-YYYY format."""
    return datetime.now().strftime("%m-%d-%Y")
