import torch
from torch import nn, optim
from torchvision import datasets, models, transforms
import argparse
import os

# Argument parser to handle command line inputs
parser = argparse.ArgumentParser(description="Train a neural network on a dataset")
parser.add_argument('data_dir', type=str, help='Dataset directory')
parser.add_argument('--save_dir', type=str, default='.', help='Directory to save checkpoints')
parser.add_argument('--arch', type=str, default='vgg16', help='Model architecture (vgg16 or resnet18)')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units in classifier')
parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
parser.add_argument('--gpu', action='store_true', help='Use GPU if available')

args = parser.parse_args()

# Set device to GPU if available and chosen
device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

# Data loading and transformations
data_dir = args.data_dir
train_dir = os.path.join(data_dir, 'train')
valid_dir = os.path.join(data_dir, 'valid')

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
}

train_data = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
valid_data = datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64)

# Load pretrained model
if args.arch == 'vgg16':
    model = models.vgg16(pretrained=True)
elif args.arch == 'resnet18':
    model = models.resnet18(pretrained=True)
else:
    print(f"Unsupported architecture {args.arch}")
    exit()

# Freeze feature parameters
for param in model.parameters():
    param.requires_grad = False

# Replace classifier
if args.arch == 'vgg16':
    input_size = 25088  # VGG16 input to classifier
elif args.arch == 'resnet18':
    input_size = 512  # ResNet18 input to classifier

classifier = nn.Sequential(
                        nn.Linear(input_size, hidden_size),
                        nn.ELU(),
                        nn.Dropout(dropout_rate),
                        nn.Linear(hidden_size, output_size)
        )
if args.arch == 'vgg16':
    model.classifier = classifier
else:
    model.fc = classifier

model.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters() if args.arch == 'vgg16' else model.fc.parameters(), lr=args.learning_rate)

# Training the network
epochs = args.epochs
steps = 0
running_loss = 0
print_every = 40

for epoch in range(epochs):
    for inputs, labels in train_loader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    valid_loss += batch_loss.item()

                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {valid_loss/len(valid_loader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(valid_loader):.3f}")
            running_loss = 0
            model.train()

# Save the checkpoint
model.class_to_idx = train_data.class_to_idx
checkpoint = {'architecture': args.arch,
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx,
              'classifier': model.classifier if args.arch == 'vgg16' else model.fc}

torch.save(checkpoint, os.path.join(args.save_dir, 'checkpoint.pth'))
