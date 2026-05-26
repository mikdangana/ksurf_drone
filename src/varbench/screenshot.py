import tempfile
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import chromedriver_autoinstaller

# Ensure ChromeDriver is installed and updated
chromedriver_autoinstaller.install()

# Create a temporary directory for a fresh user session
temp_dir = tempfile.mkdtemp()

chrome_options = Options()
chrome_options.add_argument(f"--user-data-dir={temp_dir}")  # Unique user profile
chrome_options.add_argument("--no-sandbox")  # Bypass sandbox issues
chrome_options.add_argument("--disable-dev-shm-usage")  # Prevent shared memory issues
chrome_options.add_argument("--headless")  # Run without UI (optional)
chrome_options.add_argument("--disable-gpu")  # Fix potential GPU conflicts
chrome_options.add_argument("--remote-debugging-port=9222")  # Avoid default profile issues

# Start WebDriver with safe options
driver = webdriver.Chrome(options=chrome_options)

# Open Spekt8 UI
driver.get("http://localhost:8080")

# Save screenshot
driver.save_screenshot("spekt8_diagram.png")

# Close browser
driver.quit()

print(f"Screenshot saved as spekt8_diagram.png using profile {temp_dir}")

