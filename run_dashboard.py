"""
Stock Dashboard Launcher
Launch the stock visualization dashboard
"""
import os
import sys
import webbrowser
from threading import Timer
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

def open_browser(port=8050):
    """Open browser to the dashboard"""
    webbrowser.open_new(f"http://localhost:{port}")

def main():
    """Launch the dashboard"""
    # Get the absolute path to the app.py file
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard", "app.py")
    
    if not os.path.exists(app_path):
        logger.error(f"Dashboard app file not found at {app_path}")
        return
    
    port = 8050
    
    # Open browser after a short delay
    Timer(3, open_browser, kwargs={"port": port}).start()
    
    # Run the dashboard application
    logger.info(f"Starting dashboard at http://localhost:{port}")
    subprocess.run([sys.executable, app_path])

if __name__ == "__main__":
    main()