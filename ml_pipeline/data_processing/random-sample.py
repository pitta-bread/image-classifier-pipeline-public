import glob
import os
import random
import datetime


# config
sample_size = 500
week_or_day = "week"
destination_directory = "data/labeling-20230606"

# get x random files from the dataset folder
files = glob.glob(
    f"data/last-{week_or_day}-tapo"
    f"-{datetime.date.today().strftime('%Y%m%d')}/*"
    )
files_sample = random.sample(files, sample_size)

# copy the files to the destination folder on a windows machine
for file in files_sample:
    os.system(f'copy "{file}" "{destination_directory}"')

# in the destination folder, rename each file so that the file extension is
# added ".jpg", but only if it's definitely a file and not a directory
for file in [f for f in glob.glob(f"{destination_directory}/*")
             if os.path.isfile(f) and not f.endswith(".jpg")]:
    os.rename(file, file + ".jpg")
