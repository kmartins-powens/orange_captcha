from PIL import Image
import os
import uuid

seen = set()

directories = ['unsorted/truck',]
for directory in directories:
	for filename in os.listdir(directory):
		im = Image.open(directory + '/' + filename)
		_id = uuid.uuid4()
		if _id in seen:
			continue
		else:
			seen.add(_id)
		im.resize((75, 75)).convert("RGB").save(f"data/tracteur/{_id}.jpg", "JPEG")
