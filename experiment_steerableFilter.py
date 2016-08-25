import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!

import train

options = {
    # global setup settings, and checkpoints
    'name': 'steerableFilter',
    'seed': 123,
    'checkpoint_output_directory': 'checkpoints',

    # model and dataset
    'dataset_file': 'datasets.dataset_steerableFilter',
    'model_file': 'models.model_steerableFilter',
    'pretrained_model_path': None,

    # training parameters
    'image_dim': 64,
    'batch_size': 64,
    'loss': 'squared_error',
    'learning_rate': 1e-3,
    'decay_after': 10,
    'num_epochs': 50,
    'batches_per_epoch': 100,
    'save_after': 10
}

modelOptions = {
    'batch_size': options['batch_size'],
    'npx': options['image_dim'],
    'input_seqlen': 2,
    'target_seqlen': 1,
    'dynamic_filter_size': (9, 9)
}
options['modelOptions'] = modelOptions

datasetOptions = {
    'batch_size': options['batch_size'],
    'image_size': options['image_dim'],
    'num_frames': modelOptions['input_seqlen'] + modelOptions['target_seqlen'],
    'mode': 'standard'
}
options['datasetOptions'] = datasetOptions

train.train(options)