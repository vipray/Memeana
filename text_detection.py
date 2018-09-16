# USAGE
# python text_detection.py --image images/lebron_james.jpg --east frozen_east_text_detection.pb

# import the necessary packages
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import time
import cv2
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter
import os,tempfile,subprocess

# masking function
def rounded_rectangle(draw, xy, rad, fill=None):
    x0, y0, x1, y1 = xy
    draw.rectangle([ (x0, y0 + rad), (x1, y1 - rad) ], fill=fill)
    draw.rectangle([ (x0 + rad, y0), (x1 - rad, y1) ], fill=fill)
    draw.pieslice([ (x0, y0), (x0 + rad * 2, y0 + rad * 2) ], 180, 270, fill=fill)
    draw.pieslice([ (x1 - rad * 2, y1 - rad * 2), (x1, y1) ], 0, 90, fill=fill)
    draw.pieslice([ (x0, y1 - rad * 2), (x0 + rad * 2, y1) ], 90, 180, fill=fill)
    draw.pieslice([ (x1 - rad * 2, y0), (x1, y0 + rad * 2) ], 270, 360, fill=fill)


def ocr(path):
        temp = tempfile.NamedTemporaryFile(delete=False)
        process = subprocess.Popen(["tesseract",path,temp.name],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,shell=True)
        process.communicate()
        with open(temp.name + ".txt" , 'r') as handle:
                contents = handle.read()

        os.remove(temp.name+".txt")
        os.remove(temp.name)
        return contents

def mask_image(filename):
    #str = ocr("./images/"+filename)
    #print(str)

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", type=str,
            help="path to input image")
    ap.add_argument("-east", "--east", type=str,
            help="path to input EAST text detector")
    ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
            help="minimum probability required to inspect a region")
    ap.add_argument("-w", "--width", type=int, default=320,
            help="resized image width (should be multiple of 32)")
    ap.add_argument("-e", "--height", type=int, default=320,
            help="resized image height (should be multiple of 32)")
    args = vars(ap.parse_args())

    # load the input image and grab the image dimensions
    image = cv2.imread("./images/"+filename)
    orig = image.copy()
    (H, W) = image.shape[:2]

    # set the new width and height and then determine the ratio in change
    # for both the width and height
    (newW, newH) = (args["width"], args["height"])
    rW = W / float(newW)
    rH = H / float(newH)

    # resize the image and grab the new image dimensions
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    # define the two output layer names for the EAST detector model that
    # we are interested -- the first is the output probabilities and the
    # second can be used to derive the bounding box coordinates of text
    layerNames = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]

    # load the pre-trained EAST text detector
    print("[INFO] loading EAST text detector...")
    net = cv2.dnn.readNet('frozen_east_text_detection.pb')

    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
            (123.68, 116.78, 103.94), swapRB=True, crop=False)
    start = time.time()
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    end = time.time()

    # show timing information on text prediction
    print("[INFO] text detection took {:.6f} seconds".format(end - start))

    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # loop over the number of rows
    for y in range(0, numRows):
            # extract the scores (probabilities), followed by the geometrical
            # data used to derive potential bounding box coordinates that
            # surround text
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]

            # loop over the number of columns
            for x in range(0, numCols):
                    # if our score does not have sufficient probability, ignore it
                    if scoresData[x] < args["min_confidence"]:
                            continue

                    # compute the offset factor as our resulting feature maps will
                    # be 4x smaller than the input image
                    (offsetX, offsetY) = (x * 4.0, y * 4.0)

                    # extract the rotation angle for the prediction and then
                    # compute the sin and cosine
                    angle = anglesData[x]
                    cos = np.cos(angle)
                    sin = np.sin(angle)

                    # use the geometry volume to derive the width and height of
                    # the bounding box
                    h = xData0[x] + xData2[x]
                    w = xData1[x] + xData3[x]

                    # compute both the starting and ending (x, y)-coordinates for
                    # the text prediction bounding box
                    endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                    endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                    startX = int(endX - w)
                    startY = int(endY - h)

                    # add the bounding box coordinates and probability score to
                    # our respective lists
                    rects.append((startX, startY, endX, endY))
                    confidences.append(scoresData[x])

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    im = Image.open("./images/"+filename)


    #mask.save('mask.png')

    # Blur image
    blurred = im.filter(ImageFilter.GaussianBlur(20))

    # Paste blurred region and save result


    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
            # scale the bounding box coordinates based on the respective
            # ratios
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)

            # draw the bounding box on the image
            cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
            # Create rounded rectangle mask
            mask = Image.new('L', im.size, 0)
            draw = ImageDraw.Draw(mask)
            rounded_rectangle(draw, (startX, startY, endX, endY), rad=0, fill=255)
            im.paste(blurred, mask=mask)
            
    im.save("./masked_images/"+filename)


mask_image('1.jpg')
