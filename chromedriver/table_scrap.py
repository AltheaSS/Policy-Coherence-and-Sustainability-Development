from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import numpy as np


# Initialize the WebDriver
driver = webdriver.Chrome()

# Navigate to the webpage
driver.get('https://www.huzaifazoom.com/hickel/sdi/table/')

# Wait for the table to load
table = WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.ID, 'rank_table'))
)

# Extract the table HTML using Selenium
table_html = table.get_attribute('outerHTML')

# Convert the table HTML to a pandas DataFrame
df = pd.read_html(table_html)[0]

# Close the browser
driver.quit()
with pd.ExcelWriter('/Users/noira/Desktop/web_scrap.xlsx', engine='openpyxl') as writer:
    df.to_excel(writer, index=False)


# Further processing or saving of df
