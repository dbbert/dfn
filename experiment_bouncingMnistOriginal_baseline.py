import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!

import train_baseline as train

options = {
    # global setup settings, and checkpoints
    'name': 'bouncingMnistOriginal_baseline',
    'seed': 123,
    'checkpoint_output_directory': 'checkpoints',

    # model and dataset
    'dataset_file': 'datasets.dataset_bouncingMnistOriginal',
    'model_file': 'models.model_bouncingMnistOriginal_baseline',
    'pretrained_model_path': None,

    # training parameters
    'image_dim': 64,
    'batch_size': 16,
    'loss': 'binary_crossentropy',
    'learning_rate': 1e-4,
    'decay_after': 20,
    'num_epochs': 100,
    'batches_per_epoch': 2 * 100,
    'save_after': 10
}

modelOptions = {
    'batch_size': options['batch_size'],
    'npx': options['image_dim'],
    'input_seqlen': 10,
    'target_seqlen': 10,
    'buffer_len': 1,
    'dynamic_filter_size': (9, 9)
}
options['modelOptions'] = modelOptions

datasetOptions = {
    'batch_size': options['batch_size'],
    'image_size': options['image_dim'],
    'num_frames': modelOptions['input_seqlen'] + modelOptions['target_seqlen'],
    'num_digits': 2,
    'background': 'zeros'
}
options['datasetOptions'] = datasetOptions

train.train(options)