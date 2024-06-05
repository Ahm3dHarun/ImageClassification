import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping



# Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_CLASSES = 10
NUM_EPOCHS = 20

# Data preparation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

imagenette_data = datasets.ImageFolder('path_to_imagenette', transform=transform)
cifar10_data = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
train_size = int(0.8 * len(imagenette_data))
val_size = len(imagenette_data) - train_size
train_data, val_data = random_split(imagenette_data, [train_size, val_size])
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
cifar10_loader = DataLoader(cifar10_data, batch_size=BATCH_SIZE, shuffle=True)



class BasicCNN(pl.LightningModule):
    def __init__(self):
        super(BasicCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 56 * 56, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_CLASSES)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=LEARNING_RATE)
    

class AllConvNet(pl.LightningModule):
    def __init__(self):
        super(AllConvNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128, NUM_CLASSES)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=LEARNING_RATE)
    

# Modified Basic CNN with Dropout
class BasicCNNWithDropout(pl.LightningModule):
    def __init__(self):
        super(BasicCNNWithDropout, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 56 * 56, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, NUM_CLASSES)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=LEARNING_RATE)



class TransferLearningModel(pl.LightningModule):
    def __init__(self, pretrained_model):
        super(TransferLearningModel, self).__init__()
        self.model = pretrained_model
        self.model.fc = nn.Linear(self.model.fc.in_features, NUM_CLASSES)  # Adjusting the final layer for CIFAR10

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=LEARNING_RATE)



# Train the model
basic_cnn = BasicCNN()
trainer = pl.Trainer(max_epochs=NUM_EPOCHS, callbacks=[EarlyStopping(monitor='val_loss')])
trainer.fit(basic_cnn, train_loader, val_loader)




# Train the model
all_conv_net = AllConvNet()
trainer = pl.Trainer(max_epochs=NUM_EPOCHS, callbacks=[EarlyStopping(monitor='val_loss')])
trainer.fit(all_conv_net, train_loader, val_loader)


# Train the model
cnn_with_dropout = BasicCNNWithDropout()
trainer = pl.Trainer(max_epochs=NUM_EPOCHS, callbacks=[EarlyStopping(monitor='val_loss')])
trainer.fit(cnn_with_dropout, train_loader, val_loader)



# Load a pre-trained model
pretrained_model = models.resnet18(pretrained=False)
pretrained_model.fc = nn.Linear(pretrained_model.fc.in_features, NUM_CLASSES)

# Train from scratch on CIFAR10
transfer_learning_model_scratch = TransferLearningModel(pretrained_model)
trainer = pl.Trainer(max_epochs=NUM_EPOCHS, callbacks=[EarlyStopping(monitor='val_loss')])
trainer.fit(transfer_learning_model_scratch, cifar10_loader, val_loader)

# Load the model with pre-trained weights from Imagenette
transfer_learning_model_pretrained = TransferLearningModel(basic_cnn)  # Assuming basic_cnn is saved and loaded

# Fine-tune on CIFAR10
trainer = pl.Trainer(max_epochs=NUM_EPOCHS, callbacks=[EarlyStopping(monitor='val_loss')])
trainer.fit(transfer_learning_model_pretrained, cifar10_loader, val_loader)



