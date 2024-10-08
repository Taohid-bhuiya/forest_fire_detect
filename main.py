import os
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights, densenet121, DenseNet121_Weights
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, precision_recall_curve
import seaborn as sns
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import average_precision_score
import pickle  # Import pickle module for saving models

def verify_images(image_paths):
    valid_paths = []
    for path in image_paths:
        try:
            with Image.open(path) as img:
                img.verify()
            valid_paths.append(path)
        except (IOError, Image.UnidentifiedImageError):
            print(f"Removing corrupted image {path}")
    return valid_paths

class FireDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

class FireDetectionCNN(nn.Module):
    def __init__(self, base_model_func, pretrained_weights):
        super(FireDetectionCNN, self).__init__()
        self.base_model = base_model_func(weights=pretrained_weights)
        if hasattr(self.base_model, 'fc'):
            self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 2)
        elif hasattr(self.base_model, 'classifier'):
            self.base_model.classifier = nn.Linear(self.base_model.classifier.in_features, 2)

    def forward(self, x):
        return self.base_model(x)

def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    accuracy = accuracy_score(y_true, y_pred)
    fpr = cm[0][1] / (cm[0][1] + cm[0][0])
    tpr = recall
    return precision, recall, f1, accuracy, fpr, tpr, cm

def display_confusion_matrix(cm, classes, model_name):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.savefig(f'{model_name}_confusion_matrix.png')
    plt.close()

class PerformanceTracker:
    def __init__(self):
        self.history = {
            'train': {'precision': [], 'recall': [], 'f1': [], 'accuracy': [], 'fpr': [], 'tpr': [], 'loss': []},
            'val': {'precision': [], 'recall': [], 'f1': [], 'accuracy': [], 'fpr': [], 'tpr': [], 'loss': []}
        }

    def update(self, phase, metrics, loss):
        self.history[phase]['precision'].append(metrics[0])
        self.history[phase]['recall'].append(metrics[1])
        self.history[phase]['f1'].append(metrics[2])
        self.history[phase]['accuracy'].append(metrics[3])
        self.history[phase]['fpr'].append(metrics[4])
        self.history[phase]['tpr'].append(metrics[5])
        if loss is not None:
            self.history[phase]['loss'].append(loss)

    def plot_metrics(self, phase, model_name):
        fig, ax = plt.subplots(3, 2, figsize=(20, 15))
        ax = ax.ravel()
        for i, metric in enumerate(['precision', 'recall', 'f1', 'accuracy', 'fpr', 'tpr']):
            ax[i].plot(range(1, len(self.history[phase][metric]) + 1), self.history[phase][metric], marker='o')
            ax[i].set_title(f'{metric.title()} over Epochs for {model_name}')
            ax[i].set_xlabel('Epochs')
            ax[i].set_ylabel(metric.title())
        plt.tight_layout()
        plt.savefig(f'{model_name}_{phase}_metrics.png')
        plt.close()

    def plot_precision_recall_curve(self, y_true, y_scores, model_name, phase):
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, marker='.', label='Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve for {model_name} - {phase}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{model_name}_{phase}_precision_recall_curve.png')
        plt.close()

class EarlyStopping:
    def __init__(self, patience=2, min_delta=0.01, monitors=['precision', 'recall', 'f1', 'accuracy', 'fpr', 'tpr']):
        self.patience = patience
        self.min_delta = min_delta
        self.monitors = monitors
        self.counter = {m: 0 for m in monitors}
        self.best_scores = {m: None for m in monitors}
        self.early_stop = {m: False for m in monitors}
        self.monitor_op = {m: (np.less if m == 'loss' else np.greater) for m in monitors}

    def __call__(self, metrics):
        for i, metric in enumerate(self.monitors):
            score = metrics[i]
            if self.best_scores[metric] is None:
                self.best_scores[metric] = score
            elif self.monitor_op[metric](score, self.best_scores[metric] + self.min_delta):
                self.best_scores[metric] = score
                self.counter[metric] = 0
            else:
                self.counter[metric] += 1
                if self.counter[metric] >= self.patience:
                    self.early_stop[metric] = True

        if all(self.early_stop.values()):
            print("Early stopping triggered for all metrics.")
            return True
        return False

def evaluate_and_visualize(model, model_name, loader, num_samples=20):
    model.eval()
    images, actuals, predictions = [], [], []
    with torch.no_grad():
        for img, labels in loader:
            img, labels = img.to(device), labels.to(device)
            outputs = model(img)
            _, preds = torch.max(outputs, 1)
            if len(images) < num_samples:
                images.extend(img.cpu())
                actuals.extend(labels.cpu())
                predictions.extend(preds.cpu())
            if len(images) >= num_samples:
                break

    plt.figure(figsize=(20, 14))
    for i in range(num_samples):
        ax = plt.subplot(4, 5, i + 1, xticks=[], yticks=[])
        img_path = loader.dataset.image_paths[i]
        image = Image.open(img_path)
        ax.imshow(image)
        actual_label = 'Fire' if actuals[i].item() == 1 else 'Non-Fire'
        predicted_label = 'Fire' if predictions[i].item() == 1 else 'Non-Fire'
        ax.set_title(f'Actual: {actual_label}, Pred: {predicted_label}')
    plt.tight_layout()
    plt.savefig(f'{model_name}_evaluation_visualization.png')
    plt.close()

def train_and_evaluate(model, model_name, train_loader, val_loader, test_loader, criterion, optimizer, scheduler, num_epochs=5):
    early_stopping = EarlyStopping(patience=2, min_delta=0.01, monitors=['precision', 'recall', 'f1', 'accuracy', 'fpr', 'tpr'])
    tracker = PerformanceTracker()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_preds, total_labels, total_scores = [], [], []
        running_train_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            scores = outputs.softmax(dim=1)[:, 1]
            total_scores.extend(scores.detach().cpu().numpy())
            total_preds.extend(predicted.cpu().numpy())
            total_labels.extend(labels.cpu().numpy())
            loss.backward()
            optimizer.step()
        
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_metrics = calculate_metrics(total_labels, total_preds)
        tracker.update('train', train_metrics, epoch_train_loss)
        print(f"{model_name} - Epoch {epoch+1}/{num_epochs} Training Metrics:")
        print_metrics(train_metrics)
        print(f"Training Loss: {epoch_train_loss:.4f}")

        # Validation phase
        model.eval()
        total_preds, total_labels, total_scores = [], [], []
        running_val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                scores = outputs.softmax(dim=1)[:, 1]
                total_scores.extend(scores.cpu().numpy())
                total_preds.extend(predicted.cpu().numpy())
                total_labels.extend(labels.cpu().numpy())
        
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_metrics = calculate_metrics(total_labels, total_preds)
        tracker.update('val', val_metrics, epoch_val_loss)
        print(f"{model_name} - Epoch {epoch+1}/{num_epochs} Validation Metrics:")
        print_metrics(val_metrics)
        print(f"Validation Loss: {epoch_val_loss:.4f}")

        scheduler.step()

        if early_stopping([val_metrics[i] for i in range(6)]):
            print("Early stopping triggered.")
            break

    tracker.plot_metrics('train', model_name)
    tracker.plot_metrics('val', model_name)

    # Testing phase
    total_preds, total_labels, total_scores = [], [], []
    running_test_loss = 0.0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_test_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            scores = outputs.softmax(dim=1)[:, 1]
            total_scores.extend(scores.cpu().numpy())
            total_preds.extend(predicted.cpu().numpy())
            total_labels.extend(labels.cpu().numpy())
    
    epoch_test_loss = running_test_loss / len(test_loader.dataset)
    test_metrics = calculate_metrics(total_labels, total_preds)
    print(f"{model_name} - Final Testing Metrics:")
    print_metrics(test_metrics)
    print(f"Testing Loss: {epoch_test_loss:.4f}")
    display_confusion_matrix(test_metrics[-1], ['non-fire', 'fire'], model_name)

    # Calculate and print AUPRC for test set
    test_auc_pr = average_precision_score(total_labels, total_scores)
    print(f"{model_name} - Test AUPRC: {test_auc_pr:.4f}")

    # Plot precision-recall curve for test data
    tracker.plot_precision_recall_curve(total_labels, total_scores, model_name, 'test')

    # Save the trained model in pickled format
    with open(f'{model_name}_model.pkl', 'wb') as f:
        pickle.dump(model.state_dict(), f)
    print(f"{model_name} model saved as {model_name}_model.pkl")

def print_metrics(metrics):
    precision, recall, f1, accuracy, fpr, tpr, cm = metrics
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, Accuracy: {accuracy:.3f}, FPR: {fpr:.3f}, TPR: {tpr:.3f}")

if __name__ == '__main__':
    # Set up the device for GPU usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the transformations for preprocessing the images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Define paths to datasets
    root_dir = os.path.dirname(os.path.abspath(__file__))
    train_fire_dir = os.path.join(root_dir, 'dataset', 'train', 'fire_images')
    train_non_fire_dir = os.path.join(root_dir, 'dataset', 'train', 'non_fire_images')
    test_fire_dir = os.path.join(root_dir, 'dataset', 'test', 'fire_images')
    test_non_fire_dir = os.path.join(root_dir, 'dataset', 'test', 'non_fire_images')

    # Verify images and collect paths
    image_extensions = ['.png', '.jpg', '.jpeg']
    train_fire_paths = verify_images([os.path.join(train_fire_dir, img) for img in os.listdir(train_fire_dir) if os.path.splitext(img)[-1].lower() in image_extensions])
    train_non_fire_paths = verify_images([os.path.join(train_non_fire_dir, img) for img in os.listdir(train_non_fire_dir) if os.path.splitext(img)[-1].lower() in image_extensions])
    test_fire_paths = verify_images([os.path.join(test_fire_dir, img) for img in os.listdir(test_fire_dir) if os.path.splitext(img)[-1].lower() in image_extensions])
    test_non_fire_paths = verify_images([os.path.join(test_non_fire_dir, img) for img in os.listdir(test_non_fire_dir) if os.path.splitext(img)[-1].lower() in image_extensions])

    # Combine training and testing paths and labels
    train_paths = train_fire_paths + train_non_fire_paths
    train_labels = [1] * len(train_fire_paths) + [0] * len(train_non_fire_paths)
    test_paths = test_fire_paths + test_non_fire_paths
    test_labels = [1] * len(test_fire_paths) + [0] * len(test_non_fire_paths)

    # Split training data into training and validation sets
    train_paths, val_paths, train_labels, val_labels = train_test_split(train_paths, train_labels, test_size=0.2, stratify=train_labels, random_state=42)

    # Prepare datasets and loaders
    train_dataset = FireDataset(train_paths, train_labels, transform)
    val_dataset = FireDataset(val_paths, val_labels, transform)
    test_dataset = FireDataset(test_paths, test_labels, transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Initialize models, criterion, optimizers, and schedulers
    resnet_model = FireDetectionCNN(resnet18, ResNet18_Weights.IMAGENET1K_V1).to(device)
    densenet_model = FireDetectionCNN(densenet121, DenseNet121_Weights.IMAGENET1K_V1).to(device)
    criterion = nn.CrossEntropyLoss()
    resnet_optimizer = optim.Adam(resnet_model.parameters(), lr=0.001, weight_decay=1e-5)
    densenet_optimizer = optim.Adam(densenet_model.parameters(), lr=0.001, weight_decay=1e-5)
    resnet_scheduler = StepLR(resnet_optimizer, step_size=2, gamma=0.9)
    densenet_scheduler = StepLR(densenet_optimizer, step_size=2, gamma=0.9)

    # Train and evaluate both models
    train_and_evaluate(densenet_model, 'DenseNet121', train_loader, val_loader, test_loader, criterion, densenet_optimizer, densenet_scheduler, num_epochs=5)
    train_and_evaluate(resnet_model, 'ResNet18', train_loader, val_loader, test_loader, criterion, resnet_optimizer, resnet_scheduler, num_epochs=5)
    
    # Visualizations for both models
    evaluate_and_visualize(densenet_model, 'DenseNet121', test_loader)
    evaluate_and_visualize(resnet_model, 'ResNet18', test_loader)
