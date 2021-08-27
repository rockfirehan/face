# Explaination Code of Project 2
## DoppelGanger Find your Celebrity Look-Alike

### solving  steps:
#### 1.Load Dlib Models for face detection and facedecriptor


```
# Path to landmarks and face recognition model files
PREDICTOR_PATH = '../resource/lib/publicdata/models/shape_predictor_68_face_landmarks.dat'
FACE_RECOGNITION_MODEL_PATH = '../resource/lib/publicdata/models/dlib_face_recognition_resnet_model_v1.dat'

# Initialize face detector, facial landmarks detector 
# and face recognizer
faceDetector = dlib.get_frontal_face_detector()
shapePredictor = dlib.shape_predictor(PREDICTOR_PATH)
faceRecognizer = dlib.face_recognition_model_v1(FACE_RECOGNITION_MODEL_PATH)
```

#### 2.Load Dataset--Celeb-mini and Name Mapping file


```
# Root folder of the dataset
faceDatasetFolder = '../resource/asnlib/publicdata/celeb_mini'
# Label -> Name Mapping file
labelMap = np.load("../resource/asnlib/publicdata/celeb_mapping.npy", allow_pickle=True).item()

```

---

##### The key and value in labelMap{}:
{'n00000001': 'A.J. Buckley', 'n00000002': 'A.R. Rahman',...}

*We find every subfolder has 5 images.
So we will create dictionary named "nameLabelMap' for each images*

---

#### 3.Creat nameLabelMap{}, the key is a full-path of image, value is the name of person


```
imagePaths = []
nameLabelMap = {}
Labels = []
celenames=[]

subfolders = []
for x in os.listdir(faceDatasetFolder):
    xpath = os.path.join(faceDatasetFolder, x)
    if os.path.isdir(xpath):
        subfolders.append(xpath)
        
for i, subfolder in enumerate(subfolders):
    for x in os.listdir(subfolder):
        xpath = os.path.join(subfolder, x)

        if x.endswith('JPEG'):
            imagePaths.append(xpath)
            Labels.append(i)
            celenames.append(labelMap[xpath.split('/')[-2]])
nameLabelMap=dict(zip(imagePaths,celenames))
```
#### 4.Processing face detection, and create face descriptor

---

***4.1 detect faces in image:***
```
index = {}
pathidx=[] # save detected face image path
i = 0
faceDescriptors = None
for imagePath in imagePaths:
    print("processing: {}".format(imagePath))
    img = cv2.imread(imagePath)
    faces = faceDetector(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))#BGR to RGB
    print("{} Face(s) found".format(len(faces)))
``` 

---

***4.2 Now process each face we found:***

```

    for k, face in enumerate(faces):
    
            #Find facial landmarks for each detected face
            shape = shapePredictor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), face)
    
            # convert landmarks from Dlib's format to list of (x, y) points
            landmarks = [(p.x, p.y) for p in shape.parts()]
    
            # Compute face descriptor using neural network defined in Dlib.
            # It is a 128D vector that describes the face in img identified by shape.
            faceDescriptor = faceRecognizer.compute_face_descriptor(img, shape)
    
            # Convert face descriptor from Dlib's format to list, then a NumPy array
            faceDescriptorList = [x for x in faceDescriptor]
            faceDescriptorNdarray = np.asarray(faceDescriptorList, dtype=np.float64)
            faceDescriptorNdarray = faceDescriptorNdarray[np.newaxis, :]
    
            # Stack face descriptors (1x128) for each face in images, as rows
            if faceDescriptors is None:
                faceDescriptors = faceDescriptorNdarray
            else:
                faceDescriptors = np.concatenate((faceDescriptors, faceDescriptorNdarray), axis=0)
    
            # save the label for this face in index. We will use it later to identify
            # person name corresponding to face descriptors stored in NumPy Array
            index[i] = nameLabelMap[imagePath]
            i += 1
            # save the fullpath for each face-detected image in same order. We will show the looklike person later.
            pathidx.append(imagePath)
            
```
***4.3 saving detection files in disk, we will load and use in test module code***

```
np.save('descriptors.npy', faceDescriptors) 
with open('index.pkl', 'wb') as f:
    cPickle.dump(index, f)
```


#### 5.Testing
##### 5.1 we should run face detection and get face descriptor in test image.
##### 5.2 compare the distances between test image and every enrolled image in descriptors.npy saved last step. 

##### 5.3 get the minimun distance's index in distances list, we don't need the value of threshold, just get the most likely person's index.

##### 5.4 get the image path through the index, and show it. 

```
# read test image
testImages = glob.glob('../resource/asnlib/publicdata/test-images/*.jpg')

for test in testImages:
    im = cv2.imread(test)
    imDlib = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    celeb_name = ""
    
    faceDetector = dlib.get_frontal_face_detector()
    shapePredictor = dlib.shape_predictor(PREDICTOR_PATH)
    faceRecognizer = dlib.face_recognition_model_v1(FACE_RECOGNITION_MODEL_PATH)
    
    #load the name of enrolled image in order
    index = np.load('index.pkl', allow_pickle=True)
    
    #load enrolled image in order
    faceDescriptorsEnrolled = np.load('descriptors.npy')
    
    #face detection in test image
    img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    faces = faceDetector(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    
    #compare test image with all enrolled image
    for face in faces:
        shape = shapePredictor(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), face)
        faceDescriptor = faceRecognizer.compute_face_descriptor(im, shape)
        faceDescriptorList = [m for m in faceDescriptor]
        faceDescriptorNdarray = np.asarray(faceDescriptorList, dtype=np.float64)
        faceDescriptorNdarray = faceDescriptorNdarray[np.newaxis, :]
        #caculate distances between test image and enrolled image
        distances = np.linalg.norm(faceDescriptorsEnrolled - faceDescriptorNdarray, axis=1)
        #find the most likely person's index
        argmin = np.argmin(distances)
        # minimum distance
        minDistance = distances[argmin] 
        # find the most likely person's name and full path
        label = index[argmin]
        #the name of most look like person.
        celeb_name=index[argmin]
        #the full path of most look like person image
        pathout=pathidx[argmin]
```

---

##### we get the full path of most look like person, just read and show it.

```
    plt.subplot(121)
    plt.imshow(imDlib)
    plt.title("test img")
    
    #TODO - display celeb image which looks like the test image instead of the black image. 
    plt.subplot(122)
    likeimg=cv2.imread(pathout)
    likeimgDlib=cv2.cvtColor(likeimg, cv2.COLOR_BGR2RGB)
    plt.imshow(likeimgDlib)
    plt.title("Celeb Look-Alike={}".format(celeb_name))
    plt.show()
```
##the results:
![image](https://user-images.githubusercontent.com/21227476/131166596-8931cfa9-c586-4d39-9cf2-7557f02dc0ba.png)
