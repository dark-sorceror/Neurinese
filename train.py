import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from trainer import CharacterRecognizingTrainer
from character_model import CharacterRecognizer, CharacterDataset

IMAGE_SIZE = 64
NUM_CLASSES = 3

DATA_ROOT = Path("./data")
MODEL_SAVE_PATH = Path("./CNN_char_model.pth")

def split_data(x, y, train = 0.8, val = 0.1):
    x_train, x_temp, y_train, y_temp = train_test_split(
        x, 
        y, 
        test_size = 1 - train, 
        random_state = 42
    )
    
    val_ratio = val / (1 - train)
    
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, 
        y_temp, 
        test_size = 1 - val_ratio, 
        random_state = 42
    )
    
    return x_train, x_val, x_test, y_train, y_val, y_test

if __name__ == "__main__":
    try:
        x_data = np.load(DATA_ROOT / "image.npy") 
        y_labels = np.load(DATA_ROOT / "label.npy")
    except FileNotFoundError:
        print("ERROR: Files not fouund")
        
        exit()
        
    x_train, x_val, x_test, y_train, y_val, y_test = split_data(x_data, y_labels)

    train_ds = CharacterDataset(
        data = x_train, 
        labels = y_train
    )
    val_ds = CharacterDataset(
        data = x_val, 
        labels = y_val
    )
    test_ds = CharacterDataset(
        data = x_test, 
        labels = y_test
    ) 

    train_loader = DataLoader(
        dataset = train_ds, 
        batch_size = 64, 
        shuffle = True, 
        num_workers = 4
    )
    val_loader = DataLoader(
        dataset = val_ds, 
        batch_size = 64, 
        num_workers = 4
    )
    test_loader = DataLoader(
        test_ds, 
        batch_size = 64, 
        num_workers = 4
    )

    model = CharacterRecognizer(num_classes = NUM_CLASSES)

    trainer = CharacterRecognizingTrainer(model) 

    x, y = next(iter(train_loader))
    
    # For easy debugging
    print(x.shape)
    print(x.min(), x.max()) 
    
    trainer.fit(
        train_loader = train_loader,
        val_loader = val_loader,
        epochs = 50,
        checkpoint_path = MODEL_SAVE_PATH
    )

    print("Training complete.")