import io
import requests
from lxml import etree
import re
from time import sleep
from PIL import Image
import uuid


class GatherOrangeImages:
	def __init__(self):
		self.seen = set()
		self.limit = 500

	def requests_captcha(self):
		headers = {
			'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/115.0',
			'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
			'Accept-Language': 'fr,fr-FR;q=0.8,en-US;q=0.5,en;q=0.3',
			'Connection': 'keep-alive',
			'Upgrade-Insecure-Requests': '1',
			'Sec-Fetch-Dest': 'document',
			'Sec-Fetch-Mode': 'navigate',
			'Sec-Fetch-Site': 'none',
			'Sec-Fetch-User': '?1',
		}
		url = "https://login.orange.fr/?service=sosh&return_url=https://www.sosh.fr/"
		return requests.get(url, headers=headers)

	def get_captcha_image(self, response):
		images_raw_url = response.xpath(
			"//div[contains(@class,'captcha_row')]/button[contains(@class, 'captcha_btn')]/@style"
		)
		if not images_raw_url:
			return

		images_url = []
		for url in images_raw_url:
			images_url.append(re.findall(r"background-image:url\((https://.*)\)", url)[0])

		assert len(images_url) == 9, "There's not the right amount of images"
		return images_url

	def download_images(self, urls):
		for url in urls:
			for _ in range(9):
				_id = uuid.uuid4()
				if _id in self.seen:
					continue
				else:
					self.seen.add(_id)
					break
			image = Image.open(io.BytesIO(requests.get(url).content))
			image.save(f"unsorted/unsorted_{_id}.jpg")
			print(f"unsorted/unsorted_{_id}.jpg")


	def process(self):
		if len(self.seen) >= self.limit:
			print("Limit reached")
			exit(0)
		response = self.requests_captcha()
		if response.status_code != 200 and response.status_code != 307:
			print(response.status_code, response.content)
			exit(1)

		if "captcha" in response.url:
			urls = self.get_captcha_image(etree.HTML(response.text))
			if urls:
				print(f"downloading 9 images, {len(self.seen)} images already downloaded")
				self.download_images(urls)
			else:
				print("No images")
		else:
			print("No captcha")


if __name__ == "__main__":
	orange = GatherOrangeImages()
	for i in range(60*3):
		orange.process() # 540 pictures per hour if 9 pictures per minutes
		print("Sleeping for 30 seconds")
		sleep(30)
