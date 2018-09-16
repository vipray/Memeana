# Memeana
A web-browser based meme quiz in which you have to guess what caption suits most with the shown picture.

- prepareJSON.py takes local images from "images" folder, extract text and create new images in "masked-images" folder that have blurred text along with "result.json" in root.
- scraper.py scrapes the data from "http://highsnobiety.com/p/best-memes-2017/" and does everything else the same as prepareJSON.py.
