from text_detection import mask_image
from ocrSpace import ocr_space_url
import random,json,os


prefxURL = 'http://hit-contributions.000webhostapp.com/images/'
images_path = r'C:\Users\tanuj\Downloads\Compressed\opencv-text-detection\Refresh\images'
masked_images_path = r'C:\Users\tanuj\Downloads\Compressed\opencv-text-detection\Refresh\masked_images'
files = os.listdir(images_path)
masked_files = set(os.listdir(masked_images_path))
file_texts = []
for file in files:
    print(file)
    mask_image(file)
    response = ocr_space_url(prefxURL+file)
    response = json.loads(response)
    print(response)
    text = response['ParsedResults'][0]['ParsedText']
    file_texts.append(text)

print(*file_texts,sep='\n')

l = []
set_texts = set(file_texts)
for i,file in enumerate(files):
    d = {}
    d['question'] = '/masked_images/'+file
    d['answers'] = [file_texts[i]]
    others = set_texts^set([file_texts[i]])
    d['answers'].extend(random.sample(others,3))
    d['answers'] = random.sample(d['answers'], 4)
    d['correct'] = d['answers'].index(file_texts[i])
    l.append(d)

d=dict()
d['a']=l
with open('result.json', 'w') as fp:
    json.dump(d, fp)

