import os
import json
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from model.CGCNN_data import collate_pool, get_train_val_test_loader, CIFData
from model.CGCNN_model import CrystalGraphConvNet, AverageMeter

class FineTune(object):
    def __init__(self,
                 root_dir,
                 save_dir,
                 log_every_n_steps,
                 eval_every_n_epochs,
                 epoch,
                 val_ratio,
                 test_ratio,
                 opti,
                 lr,
                 momentum,
                 weight_decay,
                 dataset_file,
                 batch_size,
                 n_conv,
                 atom_fea_len,
                 h_fea_len,
                 n_h,
                 n_out=1,
                 random_seed=1226,
                 pin_memory=False,
                 num_workers=0
                 ):
        self.lr = lr
        self.opti = opti
        self.epochs = epoch
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.data = {}
        ori_data = json.load(open(dataset_file))
        for cifname in ori_data:
            if ori_data[cifname]["has_error"] == 1:
                self.data[cifname] = 0
            elif ori_data[cifname]["has_error"] == 0:
                self.data[cifname] = 1
        self.n_conv = n_conv
        self.atom_fea_len = atom_fea_len
        self.h_fea_len = h_fea_len
        self.n_h = n_h
        self.n_out = n_out
        self.save_dir = save_dir
        self.root_dir = root_dir
        self.momentum = momentum
        collate_fn = collate_pool
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.random_seed = random_seed
        self.weight_decay = weight_decay
        self.log_every_n_steps = log_every_n_steps
        self.eval_every_n_epochs = eval_every_n_epochs
        
        self.criterion = nn.BCEWithLogitsLoss()
        
        self.dataset = CIFData(root_dir=self.root_dir, data_file=self.data, max_num_nbr=12, 
                               radius=8, dmin=0, step=0.2, random_seed=self.random_seed)
        self.device = self._get_device()
        self.model_checkpoints_folder = save_dir
        self.train_loader, self.valid_loader, self.test_loader = get_train_val_test_loader(
            dataset=self.dataset,
            random_seed=self.random_seed,
            collate_fn=collate_fn,
            pin_memory=self.pin_memory,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio
        )

    def _get_device(self):
        if torch.cuda.is_available():
            device = 'cuda'
            torch.cuda.set_device(0)
        else:
            device = 'cpu'
        print("Running on:", device)
        return device

    def calculate_metrics(self, predictions, targets):
        preds = (predictions > 0.5).astype(int)
        targets = targets.astype(int)
        
        acc = accuracy_score(targets, preds)
        f1 = f1_score(targets, preds, average='binary', zero_division=0)
        recall = recall_score(targets, preds, average='binary', zero_division=0)
        precision = precision_score(targets, preds, average='binary', zero_division=0)
        
        return acc, f1, recall, precision

    def train(self):
        structures, _, _ = self.dataset[0]
        orig_atom_fea_len = structures[0].shape[-1]
        nbr_fea_len = structures[1].shape[-1]
        
        model = CrystalGraphConvNet(
                                    orig_atom_fea_len,
                                    nbr_fea_len,
                                    atom_fea_len=self.atom_fea_len,
                                    n_conv=self.n_conv,
                                    h_fea_len=self.h_fea_len,
                                    n_h=self.n_h,
                                    n_out=self.n_out
                                    )
        
        if self.device == 'cuda':
            torch.cuda.set_device(0)
            model.to(self.device)
            print("Use cuda for torch")
        else:
            print("Only use cpu for torch")
        
        layer_list = []
        for name, _ in model.named_parameters():
            if 'fc_out' in name:
                print(name, 'new layer')
                layer_list.append(name)
        
        params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in layer_list, model.named_parameters()))))
        base_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] not in layer_list, model.named_parameters()))))
        
        if self.opti == 'SGD':
            optimizer = optim.SGD(
                [{'params': base_params, 'lr': self.lr}, {'params': params}],
                self.lr, momentum=self.momentum, 
                weight_decay=self.weight_decay)
        elif self.opti == 'Adam':
            lr_multiplier = 0.2
            optimizer = optim.Adam([{'params': base_params, 'lr': self.lr*lr_multiplier}, {'params': params}],
                                   self.lr, weight_decay=self.weight_decay)
        else:
            raise NameError('Only SGD or Adam is allowed as optimizer')        
        
        n_iter = 0
        best_valid_acc = 0
        
        for epoch_counter in range(self.epochs):
            all_preds = []
            all_targets = []
            
            model.train()
            for bn, (input, target, _) in enumerate(self.train_loader):
                if self.device == 'cuda':
                    input_var = (input[0].to(self.device, non_blocking=True),
                                 input[1].to(self.device, non_blocking=True),
                                 input[2].to(self.device, non_blocking=True),
                                 [crys_idx.to(self.device, non_blocking=True) for crys_idx in input[3]])
                    target_var = target.float()
                    if target_var.dim() == 1:
                        target_var = target_var.unsqueeze(1)

                else:
                    input_var = (input[0], input[1], input[2], input[3])
                    target_var = target.float().unsqueeze(1)
                    if target_var.dim() == 1:
                        target_var = target_var.unsqueeze(1)
                output = model(*input_var)
                loss = self.criterion(output, target_var)
                
                pred_probs = torch.sigmoid(output).detach().cpu().numpy()
                all_preds.extend(pred_probs.flatten())
                all_targets.extend(target.numpy())
                
                if bn % self.log_every_n_steps == 0:
                    print('Epoch: %d, Batch: %d, Loss: %.4f' % (epoch_counter+1, bn, loss.item()))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                n_iter += 1
            
            acc, f1, recall, precision = self.calculate_metrics(np.array(all_preds), np.array(all_targets))
            print('Epoch {} Train: Acc {:.4f}, F1 {:.4f}, Recall {:.4f}, Precision {:.4f}'.format(
                epoch_counter+1, acc, f1, recall, precision))
            
            if epoch_counter % self.eval_every_n_epochs == 0:
                valid_acc, valid_f1, valid_recall, valid_precision = self._validate(model, self.valid_loader, epoch_counter)
                if valid_acc > best_valid_acc:
                    best_valid_acc = valid_acc
                    torch.save(model.state_dict(), os.path.join(self.model_checkpoints_folder, 'model.pth'))
                    print(f'Best model saved with validation accuracy: {best_valid_acc:.4f}')
        
        self.model = model
    
    def _validate(self, model, valid_loader, n_epoch):
        all_preds = []
        all_targets = []
        losses = AverageMeter()
        
        with torch.no_grad():
            model.eval()
            for bn, (input, target, _) in enumerate(valid_loader):
                if self.device == 'cuda':
                    input_var = (input[0].to(self.device, non_blocking=True),
                                 input[1].to(self.device, non_blocking=True),
                                 input[2].to(self.device, non_blocking=True),
                                 [crys_idx.to(self.device, non_blocking=True) for crys_idx in input[3]])
                    target_var = target.to(self.device, non_blocking=True).float().unsqueeze(1)
                    if target_var.dim() == 1:
                        target_var = target_var.unsqueeze(1)
                else:
                    input_var = (input[0], input[1], input[2], input[3])
                    target_var = target.float().unsqueeze(1)
                    if target_var.dim() == 1:
                        target_var = target_var.unsqueeze(1)
                
                output = model(*input_var)
                loss = self.criterion(output, target_var)
         
                pred_probs = torch.sigmoid(output).cpu().numpy()
                all_preds.extend(pred_probs.flatten())
                all_targets.extend(target.numpy())
                
                losses.update(loss.data.cpu().item(), target.size(0))
        
        model.train()

        acc, f1, recall, precision = self.calculate_metrics(np.array(all_preds), np.array(all_targets))
        
        print('Epoch {} Validate: Loss {:.4f}, Acc {:.4f}, F1 {:.4f}, Recall {:.4f}, Precision {:.4f}'.format(
            n_epoch+1, losses.avg, acc, f1, recall, precision))
        
        return acc, f1, recall, precision

    def test(self):
        model_path = os.path.join(self.model_checkpoints_folder, 'model.pth')
        print(model_path)
        state_dict = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(state_dict)
        
        all_preds = []
        all_targets = []
        losses = AverageMeter()
        
        with torch.no_grad():
            self.model.eval()
            for bn, (input, target, _) in enumerate(self.test_loader):
                if self.device == 'cuda':
                    input_var = (input[0].to(self.device, non_blocking=True),
                                 input[1].to(self.device, non_blocking=True),
                                 input[2].to(self.device, non_blocking=True),
                                 [crys_idx.to(self.device, non_blocking=True) for crys_idx in input[3]])
                    target_var = target.to(self.device, non_blocking=True).float().unsqueeze(1)
                    if target_var.dim() == 1:
                        target_var = target_var.unsqueeze(1)
                else:
                    input_var = (input[0], input[1], input[2], input[3])
                    target_var = target.float().unsqueeze(1)
                    if target_var.dim() == 1:
                        target_var = target_var.unsqueeze(1)
                
                output = self.model(*input_var)
                loss = self.criterion(output, target_var)
                
                pred_probs = torch.sigmoid(output).cpu().numpy()
                all_preds.extend(pred_probs.flatten())
                all_targets.extend(target.numpy())
                
                losses.update(loss.data.cpu().item(), target.size(0))
        
        acc, f1, recall, precision = self.calculate_metrics(np.array(all_preds), np.array(all_targets))
        
        preds_binary = (np.array(all_preds) > 0.5).astype(int)
        cm = confusion_matrix(np.array(all_targets).astype(int), preds_binary)
        
        print('\n' + '='*50)
        print('Test Results:')
        print('='*50)
        print(f'Loss: {losses.avg:.4f}')
        print(f'Accuracy: {acc:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'Precision: {precision:.4f}')
        print('\nConfusion Matrix:')
        print(cm)
        print('='*50 + '\n')
        
        return losses.avg, acc, f1, recall, precision
    
    def predict(self):
        model_path = os.path.join(self.model_checkpoints_folder, 'model.pth')
        state_dict = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(state_dict)
        
        with torch.no_grad():
            self.model.eval()
            
            for _, (input, target, batch_cif_ids) in enumerate(self.train_loader):
                if self.device == 'cuda':
                    input_var = (input[0].to(self.device, non_blocking=True),
                                 input[1].to(self.device, non_blocking=True),
                                 input[2].to(self.device, non_blocking=True),
                                 [crys_idx.to(self.device, non_blocking=True) for crys_idx in input[3]])
                else:
                    input_var = (input[0], input[1], input[2], input[3])
                
                output = self.model(*input_var)
                output = torch.sigmoid(output).cpu()
                
                with open(os.path.join(self.save_dir, 'train.txt'), 'a+') as f:
                    for t, o, id in zip(target, output, batch_cif_ids):
                        pred_label = 1 if o.item() > 0.5 else 0
                        line = f"{id}, true_label: {int(t.item())}, pred_prob: {o.item():.4f}, pred_label: {pred_label}\n"
                        f.write(line)

            for _, (input, target, batch_cif_ids) in enumerate(self.valid_loader):
                if self.device == 'cuda':
                    input_var = (input[0].to(self.device, non_blocking=True),
                                 input[1].to(self.device, non_blocking=True),
                                 input[2].to(self.device, non_blocking=True),
                                 [crys_idx.to(self.device, non_blocking=True) for crys_idx in input[3]])
                else:
                    input_var = (input[0], input[1], input[2], input[3])
                
                output = self.model(*input_var)
                output = torch.sigmoid(output).cpu()
                
                with open(os.path.join(self.save_dir, 'val.txt'), 'a+') as f:
                    for t, o, id in zip(target, output, batch_cif_ids):
                        pred_label = 1 if o.item() > 0.5 else 0
                        line = f"{id}, true_label: {int(t.item())}, pred_prob: {o.item():.4f}, pred_label: {pred_label}\n"
                        f.write(line)
            
            for _, (input, target, batch_cif_ids) in enumerate(self.test_loader):
                if self.device == 'cuda':
                    input_var = (input[0].to(self.device, non_blocking=True),
                                 input[1].to(self.device, non_blocking=True),
                                 input[2].to(self.device, non_blocking=True),
                                 [crys_idx.to(self.device, non_blocking=True) for crys_idx in input[3]])
                else:
                    input_var = (input[0], input[1], input[2], input[3])
                
                output = self.model(*input_var)
                output = torch.sigmoid(output).cpu()
                
                with open(os.path.join(self.save_dir, 'test.txt'), 'a+') as f:
                    for t, o, id in zip(target, output, batch_cif_ids):
                        pred_label = 1 if o.item() > 0.5 else 0
                        line = f"{id}, true_label: {int(t.item())}, pred_prob: {o.item():.4f}, pred_label: {pred_label}\n"
                        f.write(line)
        
        return "success predict"