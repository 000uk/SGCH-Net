from tqdm import tqdm
import torch
from sklearn.metrics import f1_score, confusion_matrix

class SGCMFATrainer:
    def __init__(self, model, train_loader, criterion, optimizer, scheduler, device):
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def train_epoch(self, epoch):     
        self.model.train()
        train_loss = 0
        total = 0
        correct = 0

        for inputs, targets in enumerate(tqdm(self.train_loader, desc = f"Epoch: {epoch+1}")):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_acc = 100. * correct / total
        train_loss /= len(self.train_loader)

        return train_acc, train_loss
    
    def validation(self, valid_loader):
        self.model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
            
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()

                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())

        val_acc = 100. * val_correct / val_total
        val_loss /= len(valid_loader)

        f1_macro = f1_score(all_labels, all_preds, average="macro")
        cm = confusion_matrix(all_labels, all_preds)

        return val_acc, val_loss, f1_macro, cm