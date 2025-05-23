from sklearn.metrics import roc_auc_score, f1_score
from torch.utils.data import TensorDataset, DataLoader
from simpleMLP import SimpleNN
import torch
import torch.nn as nn
import torch.optim as optim
import ast

class Evaluator:
    def __init__(self, 
                training_data, 
                test_data, 
                validation_data,
                epochs,
                fp_colname,
                class_colname
                ):
        self.training_data = training_data
        self.test_data = test_data
        self.validation_data = validation_data
        self.epochs = epochs
        self.fp_colname = fp_colname
        self.class_colname = class_colname


    def df_to_tensor(self, data):
        data[self.fp_colname] = data[self.fp_colname].apply(ast.literal_eval)
        X = torch.tensor(data[self.fp_colname].tolist(), dtype=torch.float32)
        y = torch.tensor(data[self.class_colname].values, dtype=torch.float32)
        return X, y
    
    def evaluate_roc_auc(self, model, dataloader):
        model.eval()
        all_preds = []
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                outputs = model(X_batch).squeeze()
                probs = outputs.cpu().numpy()
                preds = (outputs > 0.5).float().cpu().numpy()
                labels = y_batch.cpu().numpy()

                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(labels)

        f1 = f1_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs)
        return f1, auc
    
    def test(self):
        X_train, y_train = self.df_to_tensor(self.training_data)
        X_val, y_val = self.df_to_tensor(self.validation_data)
        X_test, y_test = self.df_to_tensor(self.test_data)

        # Wrap into DataLoaders
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

        # Initialise model
        input_dim = X_train.shape[1]
        model = SimpleNN(input_dim)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # Train the model       
        for epoch in range(self.epochs):
            model.train()
            total_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


        f1_val, auc_val = self.evaluate_roc_auc(model, val_loader)
        f1_test, auc_test = self.evaluate_roc_auc(model, test_loader)

        dct = {"f1_val": f1_val,
               "auc_val": auc_val,
               "f1_test": f1_test,
               "auc_test": auc_test}

        return dct