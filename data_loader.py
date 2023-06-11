import cv2
import os   
import pandas as pd        

class DataLoader():
    def __init__(self) -> None:
        self.images = []
        self.train = None
        self.test = None
        self.load_images()
        self.load_csv()

    def load_images(self):
        image_path = "images"
        file_list = sorted(os.listdir(image_path))
        for image_name in file_list:
            file_path = os.path.join(image_path, image_name)
            if file_path.endswith('.png'):
                self.images.append(cv2.imread(file_path))

    def load_csv(self):
        df = pd.read_csv("data.csv", sep=';')
        df = df.sort_values(by=['filename'])
        data = df.iloc[:,1:]
        data.insert(0, "data", self.images, True)
        
        # shuffle the DataFrame rows
        data = data.sample(frac = 1)

        self.train = data.iloc[500:, :]
        self.test = data.iloc[:500, :]
        return  self.train, self.test
    

