import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm

from src.datasets import ThingsMEGDataset
from src.models import BasicConvClassifier, TransformerEncoderModel, BidirectionalRNN
from src.utils import set_seed


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")

    # ------------------
    #    Dataloader
    # ------------------
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    
    train_set = ThingsMEGDataset("train", args.data_dir)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, pin_memory=True, **loader_args)
    val_set = ThingsMEGDataset("val", args.data_dir)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, pin_memory=True, **loader_args)
    test_set = ThingsMEGDataset("test", args.data_dir)
    test_loader = torch.utils.data.DataLoader(
        test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers,pin_memory=True 
    )

    # ------------------
    #       Model
    # ------------------
    """model = BidirectionalRNN(
        train_set.num_classes, train_set.num_channels#, train_set.seq_len
    ).to(args.device)"""
    model = BasicConvClassifier(
        train_set.num_classes, train_set.seq_len, train_set.num_channels
    ).to(args.device)
    print(model)
    def count_parameters(model):
        return sum(param.numel() for param in model.parameters() if param.requires_grad)

    # 表示
    print(f"Total trainable parameters: {count_parameters(model)}")
    #exit()

    # ------------------
    #     Optimizer
    # ------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-3)
    #optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)#, weight_decay=1e-5)

    # ------------------
    #   Start training
    # ------------------  
    max_val_acc = 0
    accuracy = Accuracy(
        task="multiclass", num_classes=train_set.num_classes, top_k=10
    ).to(args.device)
    
    torch.backends.cudnn.benchmark = True
    #torch.backends.cuda.enable_flash_sdp(True)
      
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc, val_loss, val_acc = [], [], [], []
        
        model.train()
        
        scaler = torch.cuda.amp.GradScaler()
        n=0
        for X, y, subject_idxs in tqdm(train_loader, desc="Train"):
            n+=1
            #input_data = input_data.to(torch.long)
            X, y = X.to(args.device, non_blocking=True), y.to(args.device, non_blocking=True)
            
            y_pred = model(X)
            
            """loss = F.cross_entropy(y_pred, y)
            train_loss.append(loss.item())"""


            '''l1 = torch.tensor(0., requires_grad=True).to(args.device)
            for w in model.parameters():
                l1 = l1 + torch.norm(w, 1)
            loss = loss + l1 *0.5'''
            
            """optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc = accuracy(y_pred, y)
            train_acc.append(acc.item())"""
            
            ## old version 量子化がうまくならないので却下
            y_pred = model(X)
            
            # Runs the forward pass with autocasting.
            with torch.cuda.amp.autocast():
                y_pred = model(X)
                loss = F.cross_entropy(y_pred, y)
            
            """l1 = torch.tensor(0., requires_grad=True).to(args.device)
            for w in model.parameters():
                l1 = l1 + torch.norm(w, 1)
            loss = loss + l1*1e-5"""

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

            train_loss.append(loss.item())
            
            acc = accuracy(y_pred, y)
            train_acc.append(acc.item())
            #if n % 100 == 0:
            #    print(train_acc,":train_acc")

        model.eval()
        for X, y, subject_idxs in tqdm(val_loader, desc="Validation"):
            X, y = X.to(args.device, non_blocking=True), y.to(args.device, non_blocking=True)
            
            with torch.no_grad():
                y_pred = model(X)
            
            val_loss.append(F.cross_entropy(y_pred, y).item())
            val_acc.append(accuracy(y_pred, y).item())

        print(f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}")
        torch.save(model.state_dict(), os.path.join(logdir, "model_last.pt"))
        if args.use_wandb:
            wandb.log({"train_loss": np.mean(train_loss), "train_acc": np.mean(train_acc), "val_loss": np.mean(val_loss), "val_acc": np.mean(val_acc)})
        
        if np.mean(val_acc) > max_val_acc:
            cprint("New best.", "cyan")
            torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
            max_val_acc = np.mean(val_acc)
            
    
    # ----------------------------------
    #  Start evaluation with best model
    # ----------------------------------
    model.load_state_dict(torch.load(os.path.join(logdir, "model_best.pt"), map_location=args.device))

    preds = [] 
    model.eval()
    for X, subject_idxs in tqdm(test_loader, desc="Validation"):        
        preds.append(model(X.to(args.device)).detach().cpu())
        
    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(logdir, "submission"), preds)
    cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")


if __name__ == "__main__":
    run()
