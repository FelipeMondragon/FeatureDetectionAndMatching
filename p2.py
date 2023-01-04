import cv2 
import numpy as np
import os

path = 'ImagesQ'
images = []
classNames = []
myList = os.listdir(path)
orb = cv2.ORB_create()
sift = cv2.SIFT_create(nfeatures=1000)


for cl in myList:
    imgCur = cv2.imread(f'{path}/{cl}',0)
    images.append(imgCur)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findIdORB(img, desList, thres = 15): 
    kp2, des2 = orb.detectAndCompute(img,None)
    bf = cv2.BFMatcher()
    matchList = []
    finalVal = -1
    try: 
        for des in desList: 
            matches = bf.knnMatch(des, des2, k=2)
            good = []
            for m,n in matches: 
                if m.distance < 0.75 * n.distance: 
                    good.append([m])
            matchList.append(len(good))
    except: 
        pass
    
    if len(matchList) != 0 :
        if max(matchList) > thres : 
            finalVal = matchList.index(max(matchList))
    return finalVal

def findIdSIFT(img, desList, thres = 15): 
    kp2, des2 = sift.detectAndCompute(img,None)
    bf = cv2.BFMatcher()
    matchList = []
    finalVal = -1
    try: 
        for des in desList: 
            matches = bf.knnMatch(des, des2, k=2)
            good = []
            for m,n in matches: 
                if m.distance < 0.75 * n.distance: 
                    good.append([m])
            matchList.append(len(good))
    except: 
        pass
    
    if len(matchList) != 0 :
        if max(matchList) > thres : 
            finalVal = matchList.index(max(matchList))
    return finalVal

def findIdHOG(img, desList, thres = 15): 
    cell_size = (16, 16)  
    block_size = (2, 2)  
    nbins = 9
    hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
                                      img.shape[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)
    hist = hog.compute(img)
    bf = cv2.BFMatcher()
    matchList = []
    finalVal = -1
    try: 
        for des in desList: 
            matches = bf.knnMatch(des, hist, k=2)
            good = []
            for m,n in matches: 
                if m.distance < 0.75 * n.distance: 
                    good.append([m])
            matchList.append(len(good))
    except: 
        pass
    
    if len(matchList) != 0 :
        if max(matchList) > thres : 
            finalVal = matchList.index(max(matchList))
    return finalVal


def findDes(images): 
    desList = []
    for img in images: 
        kp, des = orb.detectAndCompute(img,None)
        desList.append(des)
    return desList

def findDesSIFT(images): 
    desList = []
    
    for img in images: 
        kp, des = sift.detectAndCompute(img,None)
        desList.append(des)
    return desList

def findDesHOG(images): 
    desList = []
    cell_size = (16, 16)  
    block_size = (2, 2)  
    nbins = 9
    for img in images:
        hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
                                      img.shape[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)
        hist = hog.compute(img)
        desList.append(hist)
    return desList


#desList = findDes(images)
#desList = findDesSIFT(images)
desList = findDesHOG(images)
print(len(desList))

cap = cv2.VideoCapture(0)

while True: 
    success, img2 = cap.read()
    imgOriginal = img2.copy()
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    #id = findIdORB(img2, desList)
    #id = findIdSIFT(img2, desList)
    id = findIdHOG(img2, desList)
    print(id)

    if id != -1 :
        cv2.putText(imgOriginal, classNames[id], (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255),2)

    cv2.imshow('img2', imgOriginal)
    cv2.waitKey(1)

