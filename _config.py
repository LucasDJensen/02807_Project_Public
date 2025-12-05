import os
from pathlib import Path
from dotenv import load_dotenv

old_env = list(os.environ.items())
load_dotenv()
new_env = list(os.environ.items())
difference = set(new_env) - set(old_env)
# print all environment variables from .env file
print("-------------------------------------------------\n"
      "----- Environment variables from .env file: -----\n"
      "-------------------------------------------------")
for key, value in difference:
    print(f"{key}={value}")

print("-------------------------------------------------")

PATH_PROJECT = Path(os.getenv("PATH_PROJECT"))  # Path to the project folder containing this file
PATH_DATA = Path(os.getenv("PATH_DATA", PATH_PROJECT / 'data/*.nc'))  # Path to the data folder containing the .nc files. Must end with *.nc to capture all .nc files in the folder.
