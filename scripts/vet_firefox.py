import time

from selenium import webdriver
from selenium.webdriver.common.by import By

options = webdriver.FirefoxOptions()

driver = webdriver.Firefox(options=options)

#  -- TODO get observations dynamically created from Kerry's code

observations = [9578116, 9548581, 9548567]

networkUrl = "https://network.satnogs.org/"
driver.get(networkUrl)
loginPage = driver.find_element(By.XPATH, "/html/body/div[2]/nav/div/ul[2]/li/a")
loginPage.click()
time.sleep(3)
username = driver.find_element(By.NAME, "username")
username.send_keys("mohammedhyder121@gmail.com")
password = driver.find_element(By.NAME, "password")
password.send_keys("Rhok-sat401")


time.sleep(10)


# baseUrl = "https://network.satnogs.org/observations/"
testUrl = "https://network.satnogs.org/observations/9520091/"
# driver.get(baseUrl + str(9578116))
driver.get(testUrl)
time.sleep(5)

driver.quit()
