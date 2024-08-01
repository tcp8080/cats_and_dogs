import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchinfo
from torchinfo import summary
import torchvision

# from tensorflow.keras.models import load_model
import numpy as np

import random
from PIL import Image
import glob
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

import numpy as np
import seaborn as sns
sns.set_theme()

# How many subprocesses will be used for data loading (higher = more)
NUM_WORKERS = os.cpu_count()

def walk_through_dir(dir_path):
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

def plot_transformed_images(image_paths, transform, n=3, seed=42):
    random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
    i = 1
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(f) 
            ax[0].set_title(f"Original \nSize: {f.size}")
            ax[0].axis("off")
            filename = f'Original_ax_{i}.png'
            plt.savefig(filename) 

            # Transform and plot image
            # Note: permute() will change shape of image to suit matplotlib 
            # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
            transformed_image = transform(f).permute(1, 2, 0) 
            ax[1].imshow(transformed_image) 
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")
            fig.suptitle(f"Class: {Path(random_image_path).parent.stem}", fontsize=16)
            filename = f'Transformed_ax_{i}.png'
            plt.savefig(filename)
            i+=1

# # Creating a CNN-based image classifier.
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer_1 = nn.Sequential(
          nn.Conv2d(3, 64, 3, padding=1),
          nn.ReLU(),
          nn.BatchNorm2d(64),
          nn.MaxPool2d(2))
        self.conv_layer_2 = nn.Sequential(
          nn.Conv2d(64, 512, 3, padding=1),
          nn.ReLU(),
          nn.BatchNorm2d(512),
          nn.MaxPool2d(2))
        self.conv_layer_3 = nn.Sequential(
          nn.Conv2d(512, 512, kernel_size=3, padding=1),
          nn.ReLU(),
          nn.BatchNorm2d(512),
          nn.MaxPool2d(2)) 
        self.classifier = nn.Sequential(
          nn.Flatten(),
          nn.Linear(in_features=512*3*3, out_features=2))
    def forward(self, x: torch.Tensor):
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_3(x)
        x = self.classifier(x)
        return x

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer):
    # Put model in train mode
    model.train()
    
    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0
    
    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)
        
        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module):
    # Put model in eval mode
    model.eval() 
    
    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0
    
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)
    
            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            
            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
            
    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

# 1. Take in various parameters required for training and test steps
def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
        #   test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5):
    
    # 2. Create empty results dictionary
    results = {"train_loss": [],
        "train_acc": []
        # "test_loss": [],
        # "test_acc": []
    }
    
    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        # test_loss, test_acc = test_step(model=model,
        #     dataloader=test_dataloader,
        #     loss_fn=loss_fn)
        
        # 4. Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            # f"test_loss: {test_loss:.4f} | "
            # f"test_acc: {test_acc:.4f}"
        )

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        # results["test_loss"].append(test_loss)
        # results["test_acc"].append(test_acc)

    # 6. Return the filled results at the end of the epochs
    return results

def get_random_image():
    folder=r"./dogs_cats/data/test/"
    file_path = folder + random.choice(os.listdir(folder))
    return file_path
    # pil_im = Image.open(file_path, 'r')
    # return pil_im

def do_train(train_dir: str):

    ######################## Model Building with Data Augmentation ###########################
    # Set image size.
    IMAGE_WIDTH = 224
    IMAGE_HEIGHT = 224
    IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)

    # Create training transform with TrivialAugment
    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor()])

    # Create testing transform (no data augmentation)
    # test_transform = transforms.Compose([
    #     transforms.Resize(IMAGE_SIZE),
    #     transforms.ToTensor()])

    # Turn image folders into Datasets
    train_data_augmented = datasets.ImageFolder(train_dir, transform=train_transform)
    # test_data_augmented = datasets.ImageFolder(test_dir, transform=test_transform)

    # Set some parameters.
    BATCH_SIZE = 32
    torch.manual_seed(42)
    
    train_dataloader_augmented = DataLoader(train_data_augmented, 
                                            batch_size=BATCH_SIZE, 
                                            shuffle=True,
                                            num_workers=NUM_WORKERS)
    # Instantiate an object.
    model = ImageClassifier().to(device)
	
	# do a test pass through of an example input size 
    summary(model, input_size=[1, 3, IMAGE_WIDTH ,IMAGE_HEIGHT])

    # Set random seeds
    torch.manual_seed(42) 
    torch.cuda.manual_seed(42)
    
    # Set number of epochs
    NUM_EPOCHS = 2 #25
    
    # Setup loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    
    # Start the timer
    from timeit import default_timer as timer 
    start_time = timer()
    
    # Train model_0 
    model_results = train(model=model,
                          train_dataloader=train_dataloader_augmented,
                        #   test_dataloader=test_dataloader_augmented,
                          optimizer=optimizer,
                          loss_fn=loss_fn,
                          epochs=NUM_EPOCHS)
    
    model_save_path = "my_model_2.pth"
    torch.save(model.state_dict(), model_save_path)
    # End the timer and print out how long it took
    end_time = timer()
    print(f"Total training time: {end_time-start_time:.3f} seconds")


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dir = "./dogs_cats/data/train"
    do_train(train_dir)

def do_predict():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pytorch_model = ImageClassifier()
    # model_path = "./dogs_cats/model/basic_cnn_model.h5"
    model_path = "./my_model_0.pth"

    pytorch_model.load_state_dict(torch.load(model_path))
    pytorch_model.eval()

    # Load in custom image and convert the tensor values to float32
    custom_image_path = get_random_image()
    print(f"random picture:{custom_image_path}")
    custom_image = torchvision.io.read_image(str(custom_image_path)).type(torch.float32)
    # Divide the image pixel values by 255 to get them between [0, 1]
    custom_image = custom_image / 255. 
    
    # Print out image data
    print(f"Custom image tensor:\n{custom_image}\n")
    print(f"Custom image shape: {custom_image.shape}\n")
    print(f"Custom image dtype: {custom_image.dtype}")
    IMAGE_WIDTH=224
    IMAGE_HEIGHT=224
    IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
    custom_image_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
    ])
    
    # Transform target image
    custom_image_transformed = custom_image_transform(custom_image)
    
    # Print out original shape and new shape
    print(f"Original shape: {custom_image.shape}")
    print(f"New shape: {custom_image_transformed.shape}")
    with torch.inference_mode():
        # Add an extra dimension to image
        custom_image_transformed_with_batch_size = custom_image_transformed.unsqueeze(dim=0)
        
        # Print out different shapes
        print(f"Custom image transformed shape: {custom_image_transformed.shape}")
        print(f"Unsqueezed custom image shape: {custom_image_transformed_with_batch_size.shape}")
        
        # Make a prediction on image with an extra dimension
        custom_image_pred = pytorch_model(custom_image_transformed.unsqueeze(dim=0).to(device))
    # Let's convert them from logits -> prediction probabilities -> prediction labels
    # Print out prediction logits
    print(f"Prediction logits: {custom_image_pred}")
    # Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    custom_image_pred_probs = torch.softmax(custom_image_pred, dim=1)
    print(f"Prediction probabilities: {custom_image_pred_probs}")
    
    # Convert prediction probabilities -> prediction labels
    custom_image_pred_label = torch.argmax(custom_image_pred_probs, dim=1)
    print(f"Prediction label: {custom_image_pred_label}")
    
    ############################################get class ######################
    # Write transform for image
    data_transform = transforms.Compose([
        # Resize the images to IMAGE_SIZE xIMAGE_SIZE 
        transforms.Resize(size=IMAGE_SIZE),
        # Flip the images randomly on the horizontal
        transforms.RandomHorizontalFlip(p=0.5), # p = probability of flip, 0.5 = 50% chance
        # Turn the image into a torch.Tensor
        transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
    ])
    # # Creating training set
    train_dir = "./dogs_cats/data/train"
    train_data = datasets.ImageFolder(root=train_dir, # target folder of images
                                      transform=data_transform, # transforms to perform on data (images)
                                      target_transform=None) # transforms to perform on labels (if necessary)
    # Get class names as a list
    class_names = train_data.classes
    print("Class names: ",class_names)
    custom_image_pred_class = class_names[custom_image_pred_label.cpu()] # put pred label to CPU, otherwise will error
    print("final result: ",custom_image_pred_class)
    
def mytest():
    print(torch.__version__)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    image_path = "./dogs_cats"
    walk_through_dir(image_path)

    train_dir = "./dogs_cats/data/train"
    test_dir = "./dogs_cats/data/test"

    ################################# Understanding the Dataset ###########################
    # Set seed
    # random.seed(42) 
    
    # # 1. Get all image paths (* means "any combination")
    # image_path_list= glob.glob(f"{image_path}/*/*/*/*.jpg")
    
    # # 2. Get random image path
    # random_image_path = random.choice(image_path_list)
    
    # # 3. Get image class from path name (the image class is the name of the directory where the image is stored)
    # image_class = Path(random_image_path).parent.stem
    
    # # 4. Open image
    # img = Image.open(random_image_path)
    # plt.imshow(img)
    # plt.axis('off')
    # plt.savefig('random_image_before.png') 
    # # plt.show()
  
    # # 5. Print metadata
    # print(f"Random image path: {random_image_path}")
    # print(f"Image class: {image_class}")
    # print(f"Image height: {img.height}") 
    # print(f"Image width: {img.width}")

    # # Turn the image into an array
    # img_as_array = np.asarray(img)
    
    # # Plot the image with matplotlib
    # plt.figure(figsize=(8, 6))
    # plt.imshow(img_as_array)
    # plt.title(f"Image class: {image_class} | Image shape: {img_as_array.shape} -> [height, width, color_channels]")
    # plt.axis(False)
    # plt.savefig('random_image_after.png') 

    ############# Transforming Data ########################
    IMAGE_WIDTH=128
    IMAGE_HEIGHT=128
    IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
    # Write transform for image
    data_transform = transforms.Compose([
        # Resize the images to IMAGE_SIZE xIMAGE_SIZE 
        transforms.Resize(size=IMAGE_SIZE),
        # Flip the images randomly on the horizontal
        transforms.RandomHorizontalFlip(p=0.5), # p = probability of flip, 0.5 = 50% chance
        # Turn the image into a torch.Tensor
        transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
    ])

    # plot_transformed_images(image_path_list, transform=data_transform, n=3)

    ############################### Loading Image Data ###################################
    # # Creating training set
    train_data = datasets.ImageFolder(root=train_dir, # target folder of images
                                      transform=data_transform, # transforms to perform on data (images)
                                      target_transform=None) # transforms to perform on labels (if necessary)
    #Creating test set
    # test_data = datasets.ImageFolder(root=test_dir, transform=data_transform)
    
    # print(f"Train data:\n{train_data}\nTest data:\n{test_data}")
    # print(f"Train data:\n{train_data}\n")
    
    # # Get class names as a list
    # class_names = train_data.classes
    # print("Class names: ",class_names)
    
    # # Can also get class names as a dict
    # class_dict = train_data.class_to_idx
    # print("Class names as a dict: ",class_dict)
    
    # # Check the lengths
    # # print("The lengths of the training and test sets: ", len(train_data), len(test_data))
    # print("The lengths of the training: ", len(train_data))
    # img, label = train_data[0][0], train_data[0][1]
    # print(f"Image tensor:\n{img}")
    # print(f"Image shape: {img.shape}")
    # print(f"Image datatype: {img.dtype}")
    # print(f"Image label: {label}")
    # print(f"Label datatype: {type(label)}")
    # plt.imshow(img.permute(1, 2, 0))
    # plt.axis('off')
    # plt.savefig('image_before_permute.png')

    # # Rearrange the order of dimensions
    # img_permute = img.permute(1, 2, 0)
    
    # # Print out different shapes (before and after permute)
    # print(f"Original shape: {img.shape} -> [color_channels, height, width]")
    # print(f"Image permute shape: {img_permute.shape} -> [height, width, color_channels]")
    
    # # Plot the image
    # plt.figure(figsize=(10, 7))
    # plt.imshow(img.permute(1, 2, 0))
    # plt.axis("off")
    # plt.title(class_names[label], fontsize=14)
    # plt.savefig('image_after_permute.png') 

    # Turn train and test Datasets into DataLoaders
    train_dataloader = DataLoader(dataset=train_data, 
                                  batch_size=1, # how many samples per batch?
                                  num_workers=NUM_WORKERS,
                                  shuffle=True) # shuffle the data?
    
    # img, label = next(iter(train_dataloader))
    # # Note that batch size will now be 1.  
    # print(f"Image shape: {img.shape} -> [batch_size, color_channels, height, width]")
    # print(f"Label shape: {label.shape}")

    ######################## Model Building with Data Augmentation ###########################
    # Set image size.
    IMAGE_WIDTH = 224
    IMAGE_HEIGHT = 224
    IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)

    # Create training transform with TrivialAugment
    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor()])

    # Create testing transform (no data augmentation)
    # test_transform = transforms.Compose([
    #     transforms.Resize(IMAGE_SIZE),
    #     transforms.ToTensor()])

    # Turn image folders into Datasets
    train_data_augmented = datasets.ImageFolder(train_dir, transform=train_transform)
    # test_data_augmented = datasets.ImageFolder(test_dir, transform=test_transform)

    # Set some parameters.
    BATCH_SIZE = 32
    torch.manual_seed(42)
    
    train_dataloader_augmented = DataLoader(train_data_augmented, 
                                            batch_size=BATCH_SIZE, 
                                            shuffle=True,
                                            num_workers=NUM_WORKERS)
    
    # test_dataloader_augmented = DataLoader(test_data_augmented, 
    #                                        batch_size=BATCH_SIZE, 
    #                                        shuffle=False, 
    #                                        num_workers=NUM_WORKERS)

    # Instantiate an object.
    model = ImageClassifier().to(device)

    # 1. Get a batch of images and labels from the DataLoader
    # img_batch, label_batch = next(iter(train_dataloader_augmented))
    
    # # 2. Get a single image from the batch and unsqueeze the image so its shape fits the model
    # img_single, label_single = img_batch[0].unsqueeze(dim=0), label_batch[0]
    # print(f"Single image shape: {img_single.shape}\n")
    
    # 3. Perform a forward pass on a single image
    # model.eval()
    # with torch.inference_mode():
    #     pred = model(img_single.to(device))
        
    # 4. Print out what's happening and convert model logits -> pred probs -> pred label
    # print(f"Output logits:\n{pred}\n")
    # print(f"Output prediction probabilities:\n{torch.softmax(pred, dim=1)}\n")
    # print(f"Output prediction label:\n{torch.argmax(torch.softmax(pred, dim=1), dim=1)}\n")
    # print(f"Actual label:\n{label_single}")

    # do a test pass through of an example input size 
    summary(model, input_size=[1, 3, IMAGE_WIDTH ,IMAGE_HEIGHT])

    # Set random seeds
    torch.manual_seed(42) 
    torch.cuda.manual_seed(42)
    
    # Set number of epochs
    NUM_EPOCHS = 10 #25
    
    # Setup loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    
    # Start the timer
    from timeit import default_timer as timer 
    start_time = timer()
    
    # Train model_0 
    model_results = train(model=model,
                          train_dataloader=train_dataloader_augmented,
                        #   test_dataloader=test_dataloader_augmented,
                          optimizer=optimizer,
                          loss_fn=loss_fn,
                          epochs=NUM_EPOCHS)
    
    model_save_path = "my_model_1.pth"
    torch.save(model.state_dict(), model_save_path)
    # End the timer and print out how long it took
    end_time = timer()
    print(f"Total training time: {end_time-start_time:.3f} seconds")