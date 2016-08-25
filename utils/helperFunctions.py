import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
import theano
import lasagne
import os

def load_model(layers, filepath):
    checkpoint = pickle.load(open(filepath, 'rb'))
    model_values = checkpoint['model_values'] # overwrite the values of model parameters
    lasagne.layers.set_all_param_values(layers, model_values)
    return layers

def save_model(layers, epoch, history_train, start_time, host, options):
    if not os.path.exists(options['checkpoint_output_directory']):
        os.makedirs(options['checkpoint_output_directory'])

    filename = '%s_%s_%s_epoch%d_train_%.3f.p' % (
        options['name'], start_time, host, epoch+1, history_train[epoch])
    filepath = os.path.join(options['checkpoint_output_directory'], filename)

    checkpoint = {}
    checkpoint['epoch'] = epoch
    checkpoint['model_values'] = lasagne.layers.get_all_param_values(layers)
    checkpoint['layers'] = layers
    checkpoint['history_train'] = history_train      
    checkpoint['options'] = options
        
    try:
        pickle.dump(checkpoint, open(filepath, "wb"))
        print 'saved checkpoint in %s' % (filepath, )
    except Exception, e: # todo be more clever here
        print 'tried to write checkpoint into %s but got error: ' % (filepath, )
        print e
        
def visualize_prediction(data, fut=None, fig=1, case_id=0, saveId=None, savefig=False):
    if saveId is None:
        saveId = case_id

    batch_size = data.shape[0]
    num_channels = data.shape[1]
    npx = data.shape[2]
    seq_length = data.shape[4]
    
    # get data
    data = data[case_id]
    
    # get reconstruction and future sequences if exist
    if fut is not None:
        if fut.ndim == 4:
            fut_length = 1
            fut = fut[..., np.newaxis]
        else:
            fut_length = fut.shape[4]

        fut = np.concatenate((np.zeros((batch_size, num_channels, npx, npx, seq_length-fut_length)), fut), axis=4)
        fut = fut[case_id]
    
    num_rows = 1
    # create figure for original sequence
    plt.figure(2*fig, figsize=(20, 4))
    plt.clf()
    for i in xrange(seq_length):
        plt.subplot(num_rows, seq_length, i+1)
        plt.imshow(data[..., i].transpose(1,2,0).squeeze(), cmap=plt.cm.gray, interpolation="nearest")
        plt.axis('off')
    if savefig:
        plt.savefig('images/%d_gt.png' % (saveId), bbox_inches='tight', pad_inches=0, transparent = True)
    plt.show()

    # create figure for reconstuction and future sequences
    plt.figure(2*fig+1, figsize=(20, 4))
    plt.clf()
    for i in xrange(seq_length):
        if fut is not None:
            plt.subplot(num_rows, seq_length, i+1)
            plt.imshow(fut[..., i].transpose(1,2,0).squeeze(), cmap=plt.cm.gray, interpolation="nearest")
        plt.axis('off')
    if savefig:
        plt.savefig('images/%d_pred.png' % (saveId), bbox_inches='tight', pad_inches=0, transparent = True)
    plt.show()
    
def visualize_flowmapStereo(pred_filter, batch, predictions, input_seqlen, npx, filter_size, case_id, saveId = None, savefig=False):
    if saveId is None:
        saveId = case_id
    max_translation = filter_size // 2
    xFilter = np.arange(-max_translation,max_translation+1)[..., np.newaxis, np.newaxis]
#     flowX = (pred_filter[case_id] * xFilter).sum(axis=0)
    flowX = (pred_filter[case_id] * xFilter).sum(axis=0)
    flowY = np.zeros(flowX.shape)
    flowMagnitude = np.sqrt(flowX*flowX + flowY*flowY)
    print("  Minimal and maximal flow magnitude: {} / {}".format(np.amin(flowMagnitude), np.amax(flowMagnitude)))
    flowMagnitude = flowMagnitude / np.amax(flowMagnitude)
    flowOrientation = (np.arctan2(flowY, flowX) + np.pi) / (2*np.pi)
    print("  Minimal and maximal orientation: {} / {}".format(np.amin(flowOrientation), np.amax(flowOrientation)))

    flowMap = np.concatenate((flowOrientation[..., np.newaxis], flowMagnitude[..., np.newaxis], np.ones(flowMagnitude.shape)[..., np.newaxis]), 2)
    from matplotlib.colors import hsv_to_rgb
    flowMap = hsv_to_rgb(flowMap)

    ### images
    plt.figure(figsize=(20, 20))
    plt.clf()

    nImages = 4
    # input image
    plt.subplot(nImages, 1, 1)
    plt.imshow(batch[case_id, :, :, :, input_seqlen - 1].squeeze(), interpolation='none', cmap='gray')
    plt.axis('off')

    # optical flow
    plt.subplot(nImages, 1, 2)
    plt.imshow(flowMap, interpolation='none', cmap='gray')
    plt.axis('off')

    # # optical flow magnitude
    # plt.subplot(5, 1, 3)
    # plt.imshow(flowMagnitude, interpolation='none', cmap=plt.get_cmap('cool'))
    # plt.axis('off')

    # predicted image
    plt.subplot(nImages, 1, 3)
    plt.imshow(predictions[case_id, :, :, :, 0].squeeze(), interpolation='none', cmap='gray')
    plt.axis('off')

    # ground truth
    plt.subplot(nImages, 1, 4)
    plt.imshow(batch[case_id, :, :, :, input_seqlen].squeeze(), interpolation='none', cmap='gray')
    plt.axis('off')

    if savefig:
        plt.savefig('images/%d_flow.png' % (saveId), bbox_inches='tight', pad_inches=0, transparent = True)

    plt.show()

        #     # ground truth circle
        #     plt.figure(figsize=(10, 4))
        #     flowX = -np.tile(np.linspace(-1,1,num=64), (64,1))
        #     flowY = flowX.T
        #     flowMagnitude = np.sqrt(flowX*flowX + flowY*flowY)
        #     flowOrientation = (np.arctan2(flowY, flowX) + np.pi) / (2*np.pi)

        #     flowMap = np.concatenate((flowOrientation[..., np.newaxis], flowMagnitude[..., np.newaxis], np.ones(flowMagnitude.shape)[..., np.newaxis]), 2)
        #     flowMap = hsv_to_rgb(flowMap)

        #     plt.figure()
        #     plt.imshow(flowMap, interpolation='none')
        #     plt.axis('off')
        #     plt.show()

def visualize_flowmap(pred_filter, batch, predictions, input_seqlen, npx, filter_size, case_id, saveId = None, savefig=False):
    if saveId is None:
        saveId = case_id
    max_translation = filter_size // 2
    xFilter = np.tile(np.tile(np.arange(-max_translation,max_translation+1), filter_size), (npx,npx,1)).transpose(2,0,1)
    yFilter = np.tile(np.tile(np.arange(-max_translation,max_translation+1), (filter_size,1)).transpose().flatten(), (npx,npx,1)).transpose(2,0,1)
    flowX = (pred_filter[case_id] * xFilter).sum(axis=0)
    flowY = (pred_filter[case_id] * yFilter).sum(axis=0)
    flowMagnitude = np.sqrt(flowX*flowX + flowY*flowY)
#     import pdb; pdb.set_trace()
#     flowMagnitude = flowMagnitude / max_translation
    print("  Minimal and maximal flow magnitude: {} / {}".format(np.amin(flowMagnitude), np.amax(flowMagnitude)))
    flowMagnitude = flowMagnitude / np.amax(flowMagnitude)
    flowOrientation = (np.arctan2(flowY, flowX) + np.pi) / (2*np.pi)
    print("  Minimal and maximal orientation: {} / {}".format(np.amin(flowOrientation), np.amax(flowOrientation)))

    flowMap = np.concatenate((flowOrientation[..., np.newaxis], flowMagnitude[..., np.newaxis], np.ones(flowMagnitude.shape)[..., np.newaxis]), 2)
    from matplotlib.colors import hsv_to_rgb
    flowMap = hsv_to_rgb(flowMap)

    ### images
    plt.figure(figsize=(20, 20))
    plt.clf()

    nImages = 4
    # input image
    plt.subplot(nImages, 1, 1)
    plt.imshow(batch[case_id, :, :, :, input_seqlen - 1].squeeze(), interpolation='none', cmap='gray')
    plt.axis('off')

    # optical flow
    plt.subplot(nImages, 1, 2)
    plt.imshow(flowMap, interpolation='none', cmap='gray')
    plt.axis('off')

    # # optical flow magnitude
    # plt.subplot(5, 1, 3)
    # plt.imshow(flowMagnitude, interpolation='none', cmap=plt.get_cmap('cool'))
    # plt.axis('off')

    # predicted image
    plt.subplot(nImages, 1, 3)
    plt.imshow(predictions[case_id, :, :, :, 0].squeeze(), interpolation='none', cmap='gray')
    plt.axis('off')

    # ground truth
    plt.subplot(nImages, 1, 4)
    plt.imshow(batch[case_id, :, :, :, input_seqlen].squeeze(), interpolation='none', cmap='gray')
    plt.axis('off')

    if savefig:
        plt.savefig('images/%d_flow.png' % (saveId), bbox_inches='tight', pad_inches=0, transparent = True)

    plt.show()

#     # ground truth circle
#     plt.figure(figsize=(10, 4))
#     flowX = -np.tile(np.linspace(-1,1,num=64), (64,1))
#     flowY = flowX.T
#     flowMagnitude = np.sqrt(flowX*flowX + flowY*flowY)
#     flowOrientation = (np.arctan2(flowY, flowX) + np.pi) / (2*np.pi)

#     flowMap = np.concatenate((flowOrientation[..., np.newaxis], flowMagnitude[..., np.newaxis], np.ones(flowMagnitude.shape)[..., np.newaxis]), 2)
#     flowMap = hsv_to_rgb(flowMap)

#     plt.figure()
#     plt.imshow(flowMap, interpolation='none')
#     plt.axis('off')
#     plt.show()