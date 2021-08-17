'''
This file will help you to get the training_data iterator 
and perform inference on test data
'''
import glob
import random
import numpy as np
import os.path
import scipy.misc
import shutil
from numpy import (amin, amax, ravel, asarray, arange, ones, newaxis,
                   transpose, iscomplexobj, uint8, issubdtype, array)
import time
import tensorflow as tf
from PIL import Image

def bytescale(data, cmin=None, cmax=None, high=255, low=0):
  
    if data.dtype == uint8:
        return data

    if high > 255:
        raise ValueError("`high` should be less than or equal to 255.")
    if low < 0:
        raise ValueError("`low` should be greater than or equal to 0.")
    if high < low:
        raise ValueError("`high` should be greater than or equal to `low`.")

    if cmin is None:
        cmin = data.min()
    if cmax is None:
        cmax = data.max()

    cscale = cmax - cmin
    if cscale < 0:
        raise ValueError("`cmax` should be larger than `cmin`.")
    elif cscale == 0:
        cscale = 1

    scale = float(high - low) / cscale
    bytedata = (data - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype(uint8)




def toimage(arr, high=255, low=0, cmin=None, cmax=None, pal=None,
            mode=None, channel_axis=None):
   
    data = asarray(arr)
    if iscomplexobj(data):
        raise ValueError("Cannot convert a complex-valued array.")
    shape = list(data.shape)
    valid = len(shape) == 2 or ((len(shape) == 3) and
                                ((3 in shape) or (4 in shape)))
    if not valid:
        raise ValueError("'arr' does not have a suitable array shape for "
                         "any mode.")
    if len(shape) == 2:
        shape = (shape[1], shape[0])  # columns show up first
        if mode == 'F':
            data32 = data.astype(np.float32)
            image = Image.frombytes(mode, shape, data32.tostring())
            return image
        if mode in [None, 'L', 'P']:
            bytedata = bytescale(data, high=high, low=low,
                                 cmin=cmin, cmax=cmax)
            image = Image.frombytes('L', shape, bytedata.tostring())
            if pal is not None:
                image.putpalette(asarray(pal, dtype=uint8).tostring())
                # Becomes a mode='P' automagically.
            elif mode == 'P':  # default gray-scale
                pal = (arange(0, 256, 1, dtype=uint8)[:, newaxis] *
                       ones((3,), dtype=uint8)[newaxis, :])
                image.putpalette(asarray(pal, dtype=uint8).tostring())
            return image
        if mode == '1':  # high input gives threshold for 1
            bytedata = (data > high)
            image = Image.frombytes('1', shape, bytedata.tostring())
            return image
        if cmin is None:
            cmin = amin(ravel(data))
        if cmax is None:
            cmax = amax(ravel(data))
        data = (data*1.0 - cmin)*(high - low)/(cmax - cmin) + low
        if mode == 'I':
            data32 = data.astype(np.uint32)
            image = Image.frombytes(mode, shape, data32.tostring())
        else:
            raise ValueError(_errstr)
        return image

    # if here then 3-d array with a 3 or a 4 in the shape length.
    # Check for 3 in datacube shape --- 'RGB' or 'YCbCr'
    if channel_axis is None:
        if (3 in shape):
            ca = np.flatnonzero(asarray(shape) == 3)[0]
        else:
            ca = np.flatnonzero(asarray(shape) == 4)
            if len(ca):
                ca = ca[0]
            else:
                raise ValueError("Could not find channel dimension.")
    else:
        ca = channel_axis

    numch = shape[ca]
    if numch not in [3, 4]:
        raise ValueError("Channel axis dimension is not valid.")

    bytedata = bytescale(data, high=high, low=low, cmin=cmin, cmax=cmax)
    if ca == 2:
        strdata = bytedata.tostring()
        shape = (shape[1], shape[0])
    elif ca == 1:
        strdata = transpose(bytedata, (0, 2, 1)).tostring()
        shape = (shape[2], shape[0])
    elif ca == 0:
        strdata = transpose(bytedata, (1, 2, 0)).tostring()
        shape = (shape[2], shape[1])
    if mode is None:
        if numch == 3:
            mode = 'RGB'
        else:
            mode = 'RGBA'

    if mode not in ['RGB', 'RGBA', 'YCbCr', 'CMYK']:
        raise ValueError(_errstr)

    if mode in ['RGB', 'YCbCr']:
        if numch != 3:
            raise ValueError("Invalid array shape for mode.")
    if mode in ['RGBA', 'CMYK']:
        if numch != 4:
            raise ValueError("Invalid array shape for mode.")

    # Here we know data and mode is correct
    image = Image.frombytes(mode, shape, strdata)
    return image
def imsave(name, arr, format=None):
    """
    Save an array as an image.
    This function is only available if Python Imaging Library (PIL) is installed.
    .. warning::
        This function uses `bytescale` under the hood to rescale images to use
        the full (0, 255) range if ``mode`` is one of ``None, 'L', 'P', 'l'``.
        It will also cast data for 2-D images to ``uint32`` for ``mode=None``
        (which is the default).
    Parameters
    ----------
    name : str or file object
        Output file name or file object.
    arr : ndarray, MxN or MxNx3 or MxNx4
        Array containing image values.  If the shape is ``MxN``, the array
        represents a grey-level image.  Shape ``MxNx3`` stores the red, green
        and blue bands along the last dimension.  An alpha layer may be
        included, specified as the last colour band of an ``MxNx4`` array.
    format : str
        Image format. If omitted, the format to use is determined from the
        file name extension. If a file object was used instead of a file name,
        this parameter should always be used.
    Examples
    --------
    Construct an array of gradient intensity values and save to file:
    >>> from scipy.misc import imsave
    >>> x = np.zeros((255, 255))
    >>> x = np.zeros((255, 255), dtype=np.uint8)
    >>> x[:] = np.arange(255)
    >>> imsave('gradient.png', x)
    Construct an array with three colour bands (R, G, B) and store to file:
    >>> rgb = np.zeros((255, 255, 3), dtype=np.uint8)
    >>> rgb[..., 0] = np.arange(255)
    >>> rgb[..., 1] = 55
    >>> rgb[..., 2] = 1 - np.arange(255)
    >>> imsave('rgb_gradient.png', rgb)
    """
    im = toimage(arr, channel_axis=2)
    if format is None:
        im.save(name)
    else:
        im.save(name, format)
    return

def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize_images(image, [160, 576])
  return image

def load_and_preprocess_image(path):
  image = tf.read_file(path)
  return preprocess_image(image)

def preprocess_image_label(img_raw):
  b_colour = tf.constant([255,0,0],dtype=tf.uint8)
  image_decode = tf.image.decode_jpeg(img_raw,channels=3)
  image_decode = tf.image.resize_images(image_decode, [160, 576])
  image_decode = tf.cast(image_decode,dtype=tf.uint8)
  gt_bg = tf.reduce_all(tf.equal(image_decode,b_colour),axis=2)
  gt_bg2 = tf.logical_not(gt_bg)
  gt_bg = tf.stack((gt_bg,gt_bg2),axis=2)

  return gt_bg

def load_and_preprocess_image_label(gt_image_file):
  img_raw = tf.read_file(gt_image_file)
  return preprocess_image_label(img_raw)

def get_training_iterator(data_folder, image_shape, num_epochs, batch_size):
	all_image_paths = sorted(glob.glob(data_folder+'/images/'+'*.png'))
	all_label_paths = sorted(glob.glob(data_folder+'/labels/'+'*.png'))
	path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
	image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=10000) 
	path_ds = tf.data.Dataset.from_tensor_slices(all_label_paths)
	label_ds = path_ds.map(load_and_preprocess_image_label, num_parallel_calls=10000)
	image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
	dataset = image_label_ds.shuffle(buffer_size=1000)
	dataset = dataset.repeat(num_epochs)
	dataset = dataset.batch(batch_size)
	iterator = dataset.make_initializable_iterator()
	return iterator

def test_images(runs_dir, data_dir, sess, logits, keep_prob, image_pl, image_shape):
	"""
	Params:
	runs_dir : directory to save output images
	data_dir : directory which contains test and training data
	sess : tensorflow session
	keep_prob : dropout probability
	image_pl : tf placeholder for input images
	image_shape : shape of the image
	"""
	output_dir = os.path.join(runs_dir, str(time.time()))
	if os.path.exists(output_dir):
		shutil.rmtree(output_dir)
	os.makedirs(output_dir)

	data_folder = os.path.join(data_dir, 'data/testing')
	all_image_paths = sorted(glob.glob(os.path.join(data_folder,'images','*.png')))
	path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
	image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=10000) 
	image_ds = image_ds.batch(1)
	image_ds = image_ds.repeat(1)
	test_initializer = image_ds.make_initializable_iterator()
	test_image = test_initializer.get_next()
	sess.run(test_initializer.initializer)
	i=0
	while True:
		try:
			image = sess.run(test_image)
			feed = {image_pl: image,
                keep_prob: 1.0}
			# Running inference
			out_softmax = sess.run([tf.nn.softmax(logits)],feed_dict=feed)
			out_softmax = out_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
			# If out_softmax > 0.5, predicted class is road
			segmentation_out = (out_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
			# Create mask based on segmentation to apply to original image
			mask_out = np.dot(segmentation_out, np.array([[0, 255, 0, 127]]))
			mask_out = toimage(mask_out, mode="RGBA")
			# mask_out = Image.fromarray(mask_out, mode="RGBA" )
			image = np.array(image).reshape(image_shape[0], image_shape[1], 3)
			print(len(image.shape))
			street_out = toimage(image)
			# street_out = Image.fromarray(image,mode="RGB")
			street_out.paste(mask_out, box=None, mask=mask_out)
			name = os.path.basename(all_image_paths[i])
			# val = np.array(street_out)
			# val.save(os.path.join(output_dir, name))
			# street_out.save(os.path.join(output_dir, name))
			imsave(os.path.join(output_dir, name), np.array(street_out))
			

			i+=1
		except tf.errors.OutOfRangeError:
			print("End of testing dataset.")
			break

		
		
		
		



