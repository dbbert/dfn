import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!

import train

options = {
    # global setup settings, and checkpoints
    'name': 'highwayDriving',
    'seed': 123,
    'checkpoint_output_directory': 'checkpoints',

    # model and dataset
    'dataset_file': 'datasets.dataset_highwayDriving',
    'model_file': 'models.model_highwayDriving',
    'pretrained_model_path': None,

    # training parameters
    'image_dim': 64,
    'batch_size': 16,
    'loss': 'squared_error',
    'learning_rate': 1e-3,
    'decay_after': 20,
    'num_epochs': 100,
    'batches_per_epoch': 2 * 100,
    'save_after': 10
}

modelOptions = {
    'batch_size': options['batch_size'],
    'npx': options['image_dim'],
    'input_seqlen': 3,
    'target_seqlen': 3,
    'buffer_len': 2,
    'dynamic_filter_size': (11, 11),
    'refinement_network': False,
    'dynamic_bias': True
}
options['modelOptions'] = modelOptions

datasetOptions = {
    'batch_size': options['batch_size'],
    'image_size': options['image_dim'],
    'num_frames': modelOptions['input_seqlen'] + modelOptions['target_seqlen']
}
options['datasetOptions'] = datasetOptions

train.train(options)
