import torch
import torch.nn.functional as F
import numpy as np
import os
from omegaconf import OmegaConf
from utils.dataset import instruction_LGO_dataset
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from utils.custom_schedulers import get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
from utils.util import set_seed, mkdir,  load_config_file
from utils.logger import setup_logger

from torch.optim import Adam, AdamW  # both are same but AdamW has a default weight decay
from model.attribute_model import linear_attribute_model, attention_attribute_model
from utils.args import parse_arguments
from info_nce import InfoNCE
import time


def get_dataloader(args, dataset, is_train=True):

    if is_train:
        sampler = RandomSampler(dataset)
        # batch_size = config.per_gpu_train_batch_size * max(1, config.n_gpu)
    else:
        sampler = SequentialSampler(dataset)
        # batch_size = config.per_gpu_eval_batch_size * max(1, config.n_gpu)
    # sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler,
                            batch_size=args.large_batch_size, num_workers=args.dataloader_worker)

    return dataloader


def train(args, train_dataset, val_dataset, model):
    '''
    Trains the model.
    '''

    # config.train_batch_size = config.per_gpu_train_batch_size * max(1, config.n_gpu)
    train_dataloader = get_dataloader(args, train_dataset, is_train=True)
    val_dataloader = get_dataloader(args, val_dataset, is_train=False)
    # total training iterations
    # t_total = len(train_dataloader) // config.gradient_accumulation_steps \
    #     * config.num_train_epochs
    t_total = len(train_dataloader)*args.epoch

    optimizer = AdamW(model.parameters(), lr=args.contrastive_lr, eps=1.0e-08, weight_decay=0.1)

    # Warmup iterations = 20% of total iterations
    num_warmup_steps = int(0.20 * t_total)
    print("num_warmup_steps: ", num_warmup_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total)

    # if config.n_gpu > 1:
    #     model = torch.nn.DataParallel(model)

    model = model.to(torch.device(args.device))
    model.train()

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epoch)
    logger.info("  Number of GPUs = %d", 1)

    logger.info("  Total optimization steps = %d", t_total)
    if scheduler:
        logger.info("  warmup steps = %d", num_warmup_steps)

    global_step, global_loss, global_acc = 0,  0.0, 0.0
    model.zero_grad()
    loss_fc = InfoNCE(negative_mode='paired')
    min_val_loss = 9999999999
    min_val_epoch = 0
    for epoch in range(int(args.epoch)):
        for step, batch in enumerate(train_dataloader):
            batch = batch.to(torch.device(args.device))
            attr_features = model(batch)
            query = attr_features[:, 0]
            positive_key = attr_features[:, 1]
            negative_key = attr_features[:, 2:]
            loss = loss_fc(query, positive_key, negative_key)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            loss.backward()

            global_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                global_step += 1
                optimizer.step()  # PYTORCH 1.x : call optimizer.step() first then scheduler.step()

                if scheduler:
                    scheduler.step()
                model.zero_grad()
                if global_step % args.logging_steps == 0:
                    val_loss = eval(args, model, val_dataloader)
                    logger.info("Epoch: {}, global_step: {}, lr: {:.8f}, loss: {:.4f} ({:.4f}) val_loss: {:.4f}".format(epoch, global_step,
                                                                                                                        optimizer.param_groups[0]["lr"], loss.item(), global_loss / global_step, val_loss)
                                )
                    if val_loss < min_val_loss:
                        val_loss = min_val_loss
                        min_val_epoch = epoch
                    model.train()
                if (args.save_steps > 0 and global_step % args.save_steps == 0) or \
                        global_step == t_total:
                    # saving checkpoint
                    save_checkpoint(args, epoch, global_step, model, optimizer)

    return global_step, global_loss / global_step, min_val_epoch, min_val_loss


def eval(args, model, val_dataloader):
    global_step, global_loss, global_acc = 0,  0.0, 0.0
    model.eval()
    loss_fc = InfoNCE(negative_mode='paired')
    with torch.no_grad():
        for step, batch in enumerate(val_dataloader):
            batch = batch.to(torch.device(args.device))
            attr_features = model(batch)
            query = attr_features[:, 0]
            positive_key = attr_features[:, 1]
            negative_key = attr_features[:, 2:]
            loss = loss_fc(query, positive_key, negative_key)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            global_loss += loss.item()
            global_step += 1
    return global_loss/global_step


def save_checkpoint(args, epoch, global_step, model, optimizer):
    '''
    Checkpointing. Saves model and optimizer state_dict() and current epoch and global training steps.
    '''
    checkpoint_path = os.path.join(args.path2saved_checkpoints, f'checkpoint_{epoch}_{global_step}.pt')
    save_num = 0
    while (save_num < 10):
        try:

            # if args.n_gpu > 1:
            if False:
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, checkpoint_path)
            else:
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, checkpoint_path)

            logger.info("Save checkpoint to {}".format(checkpoint_path))
            break
        except:
            save_num += 1
    if save_num == 10:
        logger.info("Failed to save checkpoint after 10 trails.")
    return


def main():


    args = parse_arguments()

    global logger
    # creating directories for saving checkpoints and logs
    time_code = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    args.path2saved_checkpoints = args.path2saved_checkpoints+"/"+args.mode+"/"+time_code
    args.path2logs = args.path2logs+"/"+args.mode+"/"+time_code
    mkdir(path=args.path2saved_checkpoints)
    mkdir(path=args.path2logs)

    logger = setup_logger("ATTRIBUTE_FEATURE", args.path2logs, 0, filename="training_logs.txt")

    args.device = "cuda:1" if torch.cuda.is_available() else "cpu"
    args.n_gpu = 1  # config.n_gpu

    # set_seed(seed=11, n_gpu=config.n_gpu)

    # model = linear_attribute_model(args)
    model = attention_attribute_model(args)
    # logger.info(f"Training/evaluation parameters {train_config}")

    # getting dataset for training
    train_dataset = instruction_LGO_dataset(args, "train")
    val_dataset = instruction_LGO_dataset(args, "val")

    # Now training
    global_step, avg_loss, min_val_epoch, min_val_loss = train(args, train_dataset, val_dataset, model)

    logger.info("Training done: total_step = %s, avg loss = %s, min_val_loss = %s, min_val_epoch = %s  ", global_step, avg_loss, min_val_loss, min_val_epoch)


if __name__ == "__main__":
    main()
