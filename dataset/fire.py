import os
import cv2
import torch.utils.data as data

class Fire(data.Dataset):
    def __init__(self,
                 data_dir : str = None,
                 transform = None,
                 is_train : bool = None) -> None:
        super().__init__()

        self.datapath = data_dir
        self.transform = transform
        self.is_train = is_train

        self.data = list()
        if self.is_train:
            for line in os.listdir(os.path.join(self.datapath, 'Train_Data', 'Fire')):
                self.data.append(
                    {
                        'filename' : os.path.join(self.datapath, 'Train_Data', 'Fire', line),
                        'label' : 1
                    })
            
            for line in os.listdir(os.path.join(self.datapath, 'Train_Data', 'Non_Fire')):
                self.data.append(
                    {
                        'filename' : os.path.join(self.datapath, 'Train_Data', 'Non_Fire', line),
                        'label' : 0
                    })

        else:
            for line in os.listdir(os.path.join(self.datapath, 'Test_Data', 'Fire')):
                self.data.append(
                    {
                        'filename' : os.path.join(self.datapath, 'Test_Data', 'Fire', line),
                        'label' : 1
                    })
            
            for line in os.listdir(os.path.join(self.datapath, 'Test_Data', 'Non_Fire')):
                self.data.append(
                    {
                        'filename' : os.path.join(self.datapath, 'Test_Data', 'Non_Fire', line),
                        'label' : 0
                    })
                
        self.dataset_size = len(self.data)
        
    def __getitem__(self, index):
        try:
            image = cv2.imread(self.data[index]['filename'], cv2.IMREAD_COLOR)
            image = cv2.resize(image, (128, 128))
        except:
            print(self.data[index]['filename'])

        label = self.data[index]['label']

        if not self.transform == None:
            image, label = self.transform(image, label)
        return image, label

    def getimage(self, index):
        image = cv2.imread(self.data[index]['filename'], cv2.IMREAD_COLOR)
        image = cv2.resize(image, (128, 128))

        return image
    
    def __len__(self):
        return self.dataset_size