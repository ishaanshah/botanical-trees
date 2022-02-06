import os
import numpy as np
import torch
import wandb
import time
import cv2
import copy

from tqdm import tqdm


def tensor_to_np(img):
    img = img.to("cpu").numpy().astype(float)
    img[0, :, :] = img[0, :, :] * 0.229 + 0.485
    img[1, :, :] = img[1, :, :] * 0.224 + 0.456
    img[2, :, :] = img[2, :, :] * 0.225 + 0.406
    img = img * 255
    img = img.astype(np.uint8)
    img = img.transpose([1, 2, 0])
    return img


def debug_export_before_forward(inputs, labels, idx):
    img = tensor_to_np(inputs[0])
    la = labels[0].to(torch.uint8).to("cpu").numpy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{idx:06}_im.png", img)
    cv2.imwrite(f"{idx:06}_la.png", la)


def iou(pred, target, num_classes=3):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (
            (pred_inds[target_inds]).long().sum().data.cpu().item()
        )  # Cast to long to prevent overflows
        union = (
            pred_inds.long().sum().data.cpu().item()
            + target_inds.long().sum().data.cpu().item()
            - intersection
        )
        if union > 0:
            ious.append(float(intersection) / float(max(union, 1)))

    return np.array(ious)


def train_model(
    dataloaders,
    criterion,
    optimizer,
    model,
    device,
    dest_dir,
    num_classes,
    num_epochs=25,
):
    since = time.time()

    val_acc_history = []

    best_model_state_dict = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    counter = 0

    for epoch in range(1, num_epochs + 1):
        print("Epoch {}/{}".format(epoch, num_epochs))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_iou_means = []
            running_classwise_iou_means = []
            max_iou = 0.0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Security, skip this iteration if the batch_size is 1
                if 1 == inputs.shape[0]:
                    print("Skipping iteration because batch_size = 1")
                    continue

                # Debug
                # debug_export_before_forward(inputs, labels, counter)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                ious = iou(preds, labels, num_classes)
                iou_mean = ious.mean()
                running_loss += loss.item() * inputs.size(0)
                running_classwise_iou_means.append(ious)
                running_iou_means.append(iou_mean)

                # Keep track of best accuracy
                if iou_mean > max_iou:
                    max_iou = iou_mean
                    best_imgs = inputs
                    best_labels = labels
                    best_preds = preds

                # Increment counter
                counter = counter + 1

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            running_iou_means = np.array(running_iou_means)
            if running_iou_means is not None:
                epoch_acc = running_iou_means.mean()
            else:
                epoch_acc = 0.0

            if running_classwise_iou_means is not None:
                classwise_epoch_acc = np.asarray(
                    running_classwise_iou_means
                ).mean(axis=0)
            else:
                classwise_epoch_acc = np.asarray([0 for i in range(num_classes)])

            # Log metrics
            print(
                "{} Loss: {:.4f} Acc: {:.4f}".format(
                    phase, epoch_loss, epoch_acc
                )
            )
            for i in range(num_classes):
                wandb.log(
                    {f"{phase}_class_{i}_acc": classwise_epoch_acc[i]},
                    commit=False,
                )
            wandb.log(
                {
                    f"{phase}_loss": epoch_loss,
                    f"{phase}_acc": epoch_acc,
                },
                commit=False,
            )

            # Log the best image
            class_dict = {0: "Background", 1: "Trunk", 2: "Leaves"}
            wandb.log(
                {
                    f"{phase}_best_img": wandb.Image(
                        tensor_to_np(best_imgs[0]),
                        masks={
                            "predictions": {
                                "mask_data": best_preds[0].to("cpu").numpy(),
                                "class_labels": class_dict,
                            },
                            "ground_truth": {
                                "mask_data": best_labels[0].to("cpu").numpy(),
                                "class_labels": class_dict,
                            },
                        },
                    )
                },
                commit=False,
            )

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_state_dict = copy.deepcopy(model.state_dict())
            if phase == "val":
                val_acc_history.append(epoch_acc)

            # Save current model every 25 epochs
            if 0 == epoch % 5:
                current_model_path = os.path.join(
                    dest_dir, f"checkpoint_{epoch:04}.pth"
                )
                print(f"Save current model : {current_model_path}")
                torch.save(model.state_dict(), current_model_path)

        print()
        wandb.log({"epoch": epoch}, commit=True)

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))

    # load best model weights
    return best_model_state_dict, val_acc_history
