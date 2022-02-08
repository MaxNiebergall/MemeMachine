# %% [markdown]
# 🇭 🇪 🇱 🇱 🇴 👋
# 
# This example shows how to use `segmentation-models-pytorch` for **binary** semantic segmentation. We will use the [The Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) (this is an adopted example from Albumentations package [docs](https://albumentations.ai/docs/examples/pytorch_semantic_segmentation/), which is strongly recommended to read, especially if you never used this package for augmentations before). 
# 
# The task will be to classify each pixel of an input image either as pet 🐶🐱 or as a background.
# 
# 
# What we are going to overview in this example:  
# 
#  - 📜 `Datasets` and `DataLoaders` preparation (with predefined dataset class).  
#  - 📦 `LightningModule` preparation: defining training, validation and test routines.  
#  - 📈 Writing `IoU` metric inside the `LightningModule` for measuring quality of segmentation.  
#  - 🐶 Results visualization.
# 
# 
# > It is expected you are familiar with Python, PyTorch and have some experience with training neural networks before!

# %%
import os
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from pytorch_lightning.callbacks import ModelCheckpoint
from optuna.integration import PyTorchLightningPruningCallback
import optuna

from pprint import pprint
from FastDataLoader import FastDataLoader

# %% [markdown]
# ## Dataset

# %% [markdown]
# In this example we will use predefined `Dataset` class for simplicity. The dataset actually read pairs of images and masks from disk and return `sample` - dictionary with keys `image`, `mask` and others (not relevant for this example).
# 
# ⚠️ **Dataset preparation checklist** ⚠️
# 
# In case you writing your own dataset, please, make sure that:
# 
# 1.   **Images** 🖼  
#     ✅   Images from dataset have **the same size**, required for packing images to a batch.  
#     ✅   Images height and width are **divisible by 32**. This step is important for segmentation, because almost all models have skip-connections between encoder and decoder and all encoders have 5 downsampling stages (2 ^ 5 = 32). Very likely you will face with error when model will try to concatenate encoder and decoder features if height or width is not divisible by 32.  
#     ✅   Images have **correct axes order**. PyTorch works with CHW order, we read images in HWC [height, width, channels], don`t forget to transpose image.
# 2.   **Masks** 🔳  
#     ✅   Masks have **the same sizes** as images.   
#     ✅   Masks have only `0` - background and `1` - target class values (for binary segmentation).  
#     ✅   Even if mask don`t have channels, you need it. Convert each mask from **HW to 1HW** format for binary segmentation (expand the first dimension).
# 
# Some of these checks are included in LightningModule below during the training.
# 
# ❗️ And the main rule: your train, validation and test sets are not intersects with each other!

# %%
from generate_training_validation_data import PILImageDataset

# %%
# init train, val, test sets
train_data_dir = 'D:/MemeMachine_ProjectData/dataset/training'
validation_data_dir = 'D:/MemeMachine_ProjectData/dataset/validation'
img_width, img_height, n_channels = 800,800, 3 #TODO change dimensions to be wider, to better support text
batch_size = 1

encoders = ["resnet18","resnet34","resnet50","resnet101","resnet152"]
arch = "FPN"
encoder_name= "efficientnet-b5"
in_channels=3
out_classes = 1


class PetModel(pl.LightningModule):

        def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
            super().__init__()
            self.model = smp.create_model(
                arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
            )

            # preprocessing parameteres for image
            params = smp.encoders.get_preprocessing_params(encoder_name)
            self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
            self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

            # for image segmentation dice loss could be the best first choice
            self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

        def forward(self, image):
            # normalize image here
            image = (image - self.mean) / self.std
            mask = self.model(image)
            return mask

        def shared_step(self, batch, stage):
            
            image = batch["image"]

            # Shape of the image should be (batch_size, num_channels, height, width)
            # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
            assert image.ndim == 4

            # Check that image dimensions are divisible by 32, 
            # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of 
            # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have 
            # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
            # and we will get an error trying to concat these features
            h, w = image.shape[2:]
            assert h % 32 == 0 and w % 32 == 0

            mask = batch["mask"]

            # Shape of the mask should be [batch_size, num_classes, height, width]
            # for binary segmentation num_classes = 1
            assert mask.ndim == 4

            # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
            assert mask.max() <= 1.0 and mask.min() >= 0

            logits_mask = self.forward(image)
            
            # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
            loss = self.loss_fn(logits_mask, mask)

            # Lets compute metrics for some threshold
            # first convert mask values to probabilities, then 
            # apply thresholding
            prob_mask = logits_mask.sigmoid()
            pred_mask = (prob_mask > 0.5).float()

            # We will compute IoU metric by two ways
            #   1. dataset-wise
            #   2. image-wise
            # but for now we just compute true positive, false positive, false negative and
            # true negative 'pixels' for each image and class
            # these values will be aggregated in the end of an epoch
            tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")
            return {
                "loss": loss,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }

        def shared_epoch_end(self, outputs, stage):
            # aggregate step metics
            tp = torch.cat([x["tp"] for x in outputs])
            fp = torch.cat([x["fp"] for x in outputs])
            fn = torch.cat([x["fn"] for x in outputs])
            tn = torch.cat([x["tn"] for x in outputs])


            # per image IoU means that we first calculate IoU score for each image 
            # and then compute mean over these scores
            per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
            
            # dataset IoU means that we aggregate intersection and union over whole dataset
            # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
            # in this particular case will not be much, however for dataset 
            # with "empty" images (images without target class) a large gap could be observed. 
            # Empty images influence a lot on per_image_iou and much less on dataset_iou.
            dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

            metrics = {
                f"{stage}_per_image_iou": per_image_iou,
                f"{stage}_dataset_iou": dataset_iou,
                f"{stage}_uncoveredPixels_fn": fn,
            }
            
            self.log_dict(metrics, prog_bar=True)

        def training_step(self, batch, batch_idx):
            return self.shared_step(batch, "train")            

        def training_epoch_end(self, outputs):
            return self.shared_epoch_end(outputs, "train")

        def validation_step(self, batch, batch_idx):
            return self.shared_step(batch, "valid")

        def validation_epoch_end(self, outputs):
            return self.shared_epoch_end(outputs, "valid")

        def test_step(self, batch, batch_idx):
            return self.shared_step(batch, "test")  

        def test_epoch_end(self, outputs):
            return self.shared_epoch_end(outputs, "test")

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.0001)

if __name__ == '__main__':
    def objective(trial):
        
        encoder_name = trial.suggest_categorical("encoder_name", encoders)

        preprocess_input = get_preprocessing_fn(encoder_name, pretrained='imagenet')

        train_dataset = PILImageDataset(train_data_dir, img_width, img_height, transform=preprocess_input)
        valid_dataset = PILImageDataset(validation_data_dir, img_width, img_height, transform=preprocess_input)
        print(f"Train size: {len(train_dataset)}")
        print(f"Valid size: {len(valid_dataset)}")

        train_dataloader = FastDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers =True, prefetch_factor=5)
        valid_dataloader = FastDataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers =True, prefetch_factor=5)

        model = PetModel(arch, encoder_name=encoder_name, in_channels=in_channels, out_classes=out_classes)
        if False:
            ckpt_path=r"C:\Users\maxan\Documents\Programming\MemeMachine\MemeMachine\TextPixelMasking\model_saves\binary_segmentation_intro\binary-segmentation_introPIL-arch=0-encoder_name=0-epoch=04-valid_dataset_iou=0.824615.ckpt"
            try:
                model = model.load_from_checkpoint(ckpt_path, encoder_name=encoder_name, arch=arch, in_channels=in_channels, out_classes=out_classes)
                print("model loaded from checkpoint", ckpt_path)
            except Exception as e:
                print("no model found at path:", ckpt_path)
                print(e)
                
        trainer = pl.Trainer(
            checkpoint_callback=False,
            logger=True,
            gpus=1, 
            max_epochs=5,
            callbacks=[PyTorchLightningPruningCallback(trial, monitor="valid_dataset_iou")],
        )

        hyperparameters = dict(encoder_name=encoder_name)
        trainer.logger.log_hyperparams(hyperparameters)

        trainer.fit(
            model, 
            train_dataloaders=train_dataloader, 
            val_dataloaders=valid_dataloader,
        )

        return trainer.callback_metrics["valid_dataset_iou"].item()


    pruner = optuna.pruners.MedianPruner(n_startup_trials=1)

    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=None, timeout=None, gc_after_trial=True, show_progress_bar=True) #go until shut down

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

