import tensorflow as tf
import os.path
import warnings
import glob
import data_utils


def load_model(sess, model_path):   # 10 points
    """
    Load Pretrained VGG Model into TensorFlow.
    sess: Tensorflow session
    model_path: Path to the vgg folder, it was provided to you
    return: return tuple of tensors (input_image, keep_prob, out_layer3, out_layer4, out_layer7)
    """
    # TODO: Implement function
    # to load the model and weights use tf.saved_model.loader.load
    # Use this tensor names to fetch the layers from pretrained vgg model:
    # input_tensor         :  'image_input:0'
    # keep_prob_tensor_name :  'keep_prob:0'
    # out_layer3_tensor    :  'layer3_out:0'
    # out_layer4_tensor    :  'layer4_out:0'
    # out_layer7_tensor    :  'layer7_out:0' 

    vgg_tag = 'vgg16'
    # TODO load the model
    input_tensor         =   # TODO 
    keep_prob_tensor     =   # TODO
    out_layer3_tensor    =   # TODO
    out_layer4_tensor    =   # TODO
    out_layer7_tensor    =   # TODO
    
    return None, None, None, None, None

def create_layers(layer3_out, layer4_out, layer7_out, num_classes):   # 10 points
    """
    Pretrained model provided is VGG
    Build a FCN-8 model. Please read the paper Fully Convolutional Networks to complete this task.
    layer3_out : Output Tensor of Layer 3 of pretrained VGG model
    layer4_out : Output Tensor of Layer 4 of pretrained VGG model
    layer7_out : Output Tensor of Layer 7 of pretrained VGG model
    return: Tensor for the last layer
    
    Hints: 
    Encoder: A pre-trained VGG16 is used as an encoder. The decoder starts from Layer 7 of VGG16.
    FCN Layer-8: The last fully connected layer of VGG16 should be replaced by a 1x1 convolution.
    FCN Layer-9: FCN Layer-8 should be upsampled 2 times to match dimensions with Layer 4 of VGG 16,
        using transposed convolution (using conv2d_transpose function). 
    After that, a skip connection should be added between Layer 4 of VGG16 and FCN Layer-9.
    FCN Layer-10: FCN Layer-9 should be upsampled 2 times to match dimensions with Layer 3 of VGG16,
        using transposed convolution.
    After that, a skip connection was added between Layer 3 of VGG 16 and FCN Layer-10.
    FCN Layer-11: FCN Layer-10 should be upsampled 4 times to match dimensions with input image size 
        so we get the actual image back and depth is equal to number of classes, using transposed convolution

    """
    # TODO: Implement function
    return None


def optimizer(network_last_layer, actual_label, learning_rate, n_classes):   # 15 points
    """
    In this function create ops for tensorflow loss and optimizer
    Params:
    network_last_layer: Tensorflow tensor for the last layer in the neural network
    actual_label: Tensorflow placeholder for the actual label of the image
    learning_rate: Tensroflow placeholder for the learning rate
    n_classes: Number of classes
    return: Tuple of (logits, training_op, crossentropy_loss)
    """
    # TODO: Implement function
    return None, None, None


def run_fcn():    # 20 points
    n_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    NUM_EPOCHS = # initialize number of epochs # We used 20 to get the results provided in the document.
    BATCH_SIZE  = # initialize with batch size for training # We used 1 to get the results provided in the document.
    #Get training data iterator 
    iterator = data_utils.get_training_iterator(os.path.join(data_dir, 'data/training'), image_shape, NUM_EPOCHS, BATCH_SIZE)
    imagebx_labelbx = iterator.get_next()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        # Path to vgg model
        vggmodel_path = os.path.join(data_dir, 'vgg_pretrained')

        # TODO: Build the Fully Convolutional Network using load_model, create_layers, and optimizer function
        
        # TODO: Train the FCN network by iterating the tf.data iterator. Check the examples provided in the document 

    
    
    # TODO: Once training is completed, use the trained model on test dataset.
    #data_utils.test_images(RUNS_DIRECTORY, DATA_DIRECTORY, sess, logits,  keep_prob, image_input, image_shape)


if __name__ == "__main__":
  run_fcn()

  
