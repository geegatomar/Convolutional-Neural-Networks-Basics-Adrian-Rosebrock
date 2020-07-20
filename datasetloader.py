import cv2
import os
import numpy as np


class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors
        if preprocessors is None:
            preprocessors = []
    

    def load(self, directoryPath, verbose=-1):
        
        imageFiles = os.listdir(directoryPath)
        data = []
        labels = []

        for (i, imageFile) in enumerate(imageFiles[:5000]):

            label = imageFile.split('.')[-3]
            imagePath = os.path.join(directoryPath, imageFile)
            image = cv2.imread(imagePath)
            
            print(i, label, image.shape)
            
            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)
            
            data.append(image)
            labels.append(label)

            if verbose > 0 and i + 1 % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1, len(imageFiles)))

        return (np.array(data), np.array(labels))

                


