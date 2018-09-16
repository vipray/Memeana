import requests,json,shutil,random
from bs4 import BeautifulSoup
from text_detection import mask_image
from ocrSpace import ocr_space_url

r = requests.get("https://www.highsnobiety.com/p/best-memes-2017/")
if r.ok:
    print("Positive response")
    soup = BeautifulSoup(r.content,'html.parser')
    images = []
    i=16
    file_texts,files = [],[]
    for fig in soup.findAll('figure',{'class':  'img-element'}):
        images.append(fig.findChildren("meta" , recursive=False)[-1]['content'])
        print(images[-1])
        response = requests.get(images[-1], stream=True)
        with open('images/'+str(i)+'.jpg', 'wb') as out_file:
            response.raw.decode_content = True
            shutil.copyfileobj(response.raw, out_file)
            print("Saved %d.jpg"%i)
        del response
        mask_image(str(i)+'.jpg')
        print(images[-1])
        response = ocr_space_url(images[-1])
        response = json.loads(response)
        print(response)
        text = response['ParsedResults'][0]['ParsedText']
        file_texts.append(text)
        files.append(str(i)+'.jpg')
        i+=1
                                 
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
    with open('scrapedResult.json', 'w') as fp:
        json.dump(d, fp)
        
    
        
    
