'''
NOTICE

This software (or technical data) was produced for the U. S. Government under contract, and is subject to the Rights in Data-General Clause 52.227-14, Alt. IV (DEC 2007) 

(C) 2021 The MITRE Corporation. All Rights Reserved.
Approved for Public Release; Distribution Unlimited. Public Release Case Number 18-0812.
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from datetime import datetime
import logging
import coloredlogs

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TerminateOnNaN, CSVLogger, \
    ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.utils import Progbar
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

# this is here for https://github.com/keras-team/keras/issues/13689 (missing key 'val_binary_accuracy')
# might go away when switching out Model.fit_generator
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()

from datasets.generic import GenericDataset
import networks
import misc
import stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--model', default="CosmeticLensNetwork", choices=["CosmeticLensNetwork", "SoftLensNetwork"],
                        help="Network Model")
    parser.add_argument('--no_weights', default=False, action='store_true',
                        help="Use no pretrained weight. This trains the whole model")
    parser.add_argument('--freeze_pretrained', default=False, action='store_true',
                        help="Only used when pretrained model weights are used. Fine-tune pretrained model weights during training.")
    parser.add_argument('--prev_checkpoint', default=None,
                        help='If provided, initialize training from the given checkpoint')

    # dataset
    parser.add_argument('--img_size', default=224, type=int, help='model input image size')
    parser.add_argument('--train_metadata_file', help='Train metadata file, used if using generic dataset', required=True)
    parser.add_argument('--validation_metadata_file', help='Validation metadata file, used if using generic dataset', required=True)
    parser.add_argument('--dataset_base', help='Top level dataset folder')
    parser.add_argument('--img_path_field', help='Field in metadata files containing the image paths. If relative, dataset_base '
                        'must also be provided to form the full paths.', default='Image')
    parser.add_argument('--label_field', help='Field in metadata files containing class label, e.g. "Contact".', default='Contacts') 
    parser.add_argument('--image_suffix', help="The image type", default='')
    parser.add_argument('--color_mode', default="grayscale", choices=["grayscale", "rgb", "rgba"],
                        help="Color mode of images: default is `rgb`")
    # training
    parser.add_argument('--batch_size', default=80, type=int, help='batch size')
    parser.add_argument('--num_epochs', default=10, type=int, help='Number of epochs')
    parser.add_argument('--patience', default=10, type=int,
                        help='EarlyStopping patience epochs; zero or less disables this')
    parser.add_argument('--lr_step_size', default=10, type=int, help='Learning rate decay step size')
    parser.add_argument('--init_lr', default=1e-3, type=float, help='Initial learning rate')
    parser.add_argument('--lr_decay_factor', default=0.75, type=float, help='Learning rate decay factor')
    parser.add_argument('--no_lr_step_decay_schedule', default=False, action='store_true', help='Use constant base learning rate rather than decaying')

    # outputs
    parser.add_argument('--results_dir', default="results", help='results directory')
    parser.add_argument('--description', default="contact-lens", help="description of training")
    # misc
    parser.add_argument('--logging_level', default="INFO", help="verbosity of logging")
    args = parser.parse_args()

    # set up logging
    misc.init_output_logging()
    coloredlogs.install(fmt='%(asctime)s %(name)s %(levelname)s %(message)s', level=args.logging_level)
    logger = logging.getLogger('training')
    result_subdir = misc.create_result_subdir(args.results_dir, args.description, cp_files=["train.py"])

    img_w = args.img_size
    img_h = args.img_size
    # color_mode: One of
    color_modes = {"grayscale": 1, "rgb": 3, "rgba": 4}
    img_shape = (img_h, img_w, color_modes[args.color_mode])
    target_size = (img_h, img_w)

    dataset = GenericDataset(train_metadata_file=args.train_metadata_file, validation_metadata_file=args.validation_metadata_file, 
                             images_folder=args.dataset_base, suffix=args.image_suffix, img_path_field=args.img_path_field)


    # ------------------------------------------------------------------------------
    # -- Preparing Data Generators for training and validation set
    # ------------------------------------------------------------------------------

    # data augmentation only for the training instances (start simple)

    pretrain_weights = not args.no_weights

    if pretrain_weights and (args.model in ['SoftLensNetwork']):
        from tensorflow.keras.applications.resnet50 import preprocess_input
        train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                           horizontal_flip=True,
                                           vertical_flip=True)
        valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        print("Using ResNet50 network preprocessing")
    else:
        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                           horizontal_flip=True,
                                           vertical_flip=True)
        valid_datagen = ImageDataGenerator(rescale=1. / 255)

    train_split = dataset.split('training')
    val_split = dataset.split('validation')

    print(f"Data Samples:\n\ttraining {len(train_split)}\n\tvalidation {len(val_split)}")

    # data generators:
    print(f"Building training generator....")
    train_generator = train_datagen.flow_from_dataframe(
        color_mode=args.color_mode,
        dataframe=train_split,
        directory=None,
        x_col=dataset.files,
        y_col=args.label_field,
        target_size=target_size,
        batch_size=args.batch_size,
        shuffle=True,
        class_mode='categorical'
    )

    print(f"Building valid generator....")
    val_generator = valid_datagen.flow_from_dataframe(
        color_mode=args.color_mode,
        dataframe=val_split,
        directory=None,
        x_col=dataset.files,
        y_col=args.label_field,
        target_size=target_size,
        batch_size=args.batch_size,
        shuffle=False,
        class_mode='categorical'
    )

    num_features = val_split[args.label_field].nunique()
    if args.no_weights:
        print(f"Not using pretrain weights ")
    assert (args.model in ['SoftLensNetwork']) or (not args.freeze_pretrained), "Freeze pretrained only implemented for SoftLensNetwork"

    if args.model in ["SoftLensNetwork"]:
        model = networks.SoftLensNetwork(img_shape=img_shape, num_features=num_features, pretrain_weights=pretrain_weights, freeze_pretrained=args.freeze_pretrained)
    elif args.model in ["CosmeticLensNetwork"]:
        model = networks.CosmeticLensNetwork(img_shape=img_shape, num_features=num_features)
    else:
        raise ValueError(f"`model` unknown {args.model}")

    print(f"Using model {args.model}")
    model.summary()

    if args.prev_checkpoint is not None:
        model = load_model(args.prev_checkpoint)
        print("Loaded model from " + args.prev_checkpoint)
    else:
        print("No previous checkpoint provided - training from scratch")
    # ------------------------------------------------------------------------------
    # -- Compile model
    # ------------------------------------------------------------------------------
    optimizer = Adam(learning_rate=args.init_lr)
    model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=optimizer)

    # ------------------------------------------------------------------------------
    # -- Checkpointing: at each epoch, the best model so far is saved
    # ------------------------------------------------------------------------------
    model_path = os.path.join(result_subdir, f"weights-{args.model}" + "-epoch{epoch:02d}-{val_accuracy:.2f}.hdf5")

    callbacks = [TerminateOnNaN()]

    checkpoint = ModelCheckpoint(
        model_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    callbacks.append(checkpoint)

    cvs_logger = CSVLogger(os.path.join(result_subdir, "training.csv"))
    callbacks.append(cvs_logger)

    def step_decay_schedule(initial_lr=args.init_lr, decay_factor=args.lr_decay_factor, step_size=args.lr_step_size):
        """Wrapper function to create a LearningRateScheduler with step decay schedule."""

        def schedule(epoch):
            return initial_lr * (decay_factor ** np.floor(epoch / step_size))

        return LearningRateScheduler(schedule)

    if not args.no_lr_step_decay_schedule:
        callbacks.append(step_decay_schedule())
    else:
        print("Not using learning rate step decay schedule")

    if args.patience > 0:
        print(f"Using EarlyStopping with patience of {args.patience}")
        early_stoppping = EarlyStopping(
            monitor="val_accuracy",
            patience=args.patience,
            restore_best_weights=True
        )
        callbacks.append(early_stoppping)

        if args.patience > args.num_epochs:
            print(f"Number of epochs of {args.num_epochs} is less than patience of {args.patience}, "
                  f"increasing number of epochs to {args.num_epochs + args.patience}")
            args.num_epochs += args.patience

    # ------------------------------------------------------------------------------
    # -- Fitting
    # ----------------------------------------------------------------------------

    start_datetime = datetime.now()
    print(f"Starting training at {start_datetime}")
    history = model.fit(
        train_generator,
        epochs=args.num_epochs,
        steps_per_epoch=len(train_generator),
        validation_data=val_generator,
        validation_steps=len(val_generator),
        max_queue_size=1,
        shuffle=True,
        callbacks=callbacks,
        verbose=1
    )
    print(f"Finished training in {datetime.now() - start_datetime}")

    # plot the training and validation loss for every epoch:
    stats.plot_model_history(history, result_subdir, model_name=args.model)

    # additional reports
    if args.patience > 0:
        model.set_weights(early_stoppping.best_weights)
    # print(val_generator.class_indices)
    val_generator.reset()   # resetting generator
    nr_batches = int(np.ceil(len(val_generator.classes) / args.batch_size))
    y_true = []
    y_pred = []
    print("Generating predictions...")
    progbar = Progbar(nr_batches)
    for i, ii in enumerate(next(val_generator) for _ in range(nr_batches)):
        y_true.append(ii[1])                 # one-hot
        y_pred.append(model.predict(ii))
        progbar.update(i)

    y_true = np.vstack(y_true).astype('int')
    y_pred = np.vstack(y_pred)
    class_labels = list(val_generator.class_indices.keys())

    # y_true : array, shape = [n_samples] or [n_samples, n_classes]
    #         True binary labels or binary label indicators.
    #         The multiclass case expects shape = [n_samples] and labels
    #         with values in ``range(n_classes)``
    # y_score : array, shape = [n_samples] or [n_samples, n_classes]
    #         The multiclass case expects shape = [n_samples, n_classes]
    #         where the scores correspond to probability estimates
    print()         # progbar seems not to do a new-line when it ends
    print('ROC AUC Score')
    print(roc_auc_score(y_true, y_pred, multi_class='ovo'))
    stats.multiclass_roc(y_true, y_pred, class_labels, result_subdir, semilogx=True)
    yhat_classes = np.argmax(y_pred, axis=1)
    y_classes = np.argmax(y_true, axis=1)

    print('Confusion Matrix')
    stats.print_confusion_matrix(y_classes, yhat_classes, labels=class_labels)

    print('Classification Report')
    # hopefully target_names come back in the correct order all the time {'Cosmetic': 0, 'No': 1, 'Yes': 2}
    print(classification_report(y_classes, yhat_classes, target_names=class_labels))

    print(f"Accuracy Score {accuracy_score(y_classes, yhat_classes) * 100}%")
    val_results = model.evaluate(val_generator, verbose=False)      # true prints out progress
    print(f'Model Validation results - Loss: {val_results[0]} - Accuracy: {val_results[1] * 100}%')

    model_name = f"trained-{args.model}-{val_results[1]:.3f}.hdf5"
    print(f"Saving model as {model_name}")
    model_path = os.path.join(result_subdir, model_name)
    model.save(model_path, overwrite=True)
