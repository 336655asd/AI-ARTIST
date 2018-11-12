# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function
import tensorflow as tf
from app.fstyle.preprocessing import preprocessing_factory
from app.fstyle import reader
from app.fstyle import model
import time
import os
"""
tf.app.flags.DEFINE_string('loss_model', 'vgg_16', 'The name of the architecture to evaluate. '
                           'You can view all the support models in nets/nets_factory.py')
tf.app.flags.DEFINE_integer('image_size', 256, 'Image size to train.')
tf.app.flags.DEFINE_string("model_file", "models/denoised_starry.ckpt-done", "")
tf.app.flags.DEFINE_string("image_file", "img/test.jpg", "")

FLAGS = tf.app.flags.FLAGS
"""

print ("start neural style")
#["starry.jpg","cubist.jpg","feathers.jpg","mosaic.jpg","udnie.jpg","wave.jpg"]
style_list=["denoised_starry.ckpt-done","cubist.ckpt-done","feathers.ckpt-done","mosaic.ckpt-done",
            "udnie.ckpt-done","wave.ckpt-done","scream.ckpt-done"]
def change_style(file_name,style_iterm = 0):
    path = os.getcwd() + '/app/fstyle/'
    # Get image's height and width.
    height = 0
    width = 0
    loss_model = 'vgg_16'
    image_size = 256
    model_file = path+"models/"+style_list[style_iterm]
    image_file = file_name
    image_result = os.getcwd() + "/app/static/art/res.jpg"
    with open(image_file, 'rb') as img:
        with tf.Session().as_default() as sess:
            if image_file.lower().endswith('png'):
                image = sess.run(tf.image.decode_png(img.read()))
            else:
                image = sess.run(tf.image.decode_jpeg(img.read()))
            height = image.shape[0]
            width = image.shape[1]
    tf.logging.info('Image size: %dx%d' % (width, height))

    with tf.Graph().as_default():
        with tf.Session().as_default() as sess:

            # Read image data.
            image_preprocessing_fn, _ = preprocessing_factory.get_preprocessing(
                loss_model,
                is_training=False)
            image = reader.get_image(image_file, height, width, image_preprocessing_fn)

            # Add batch dimension
            image = tf.expand_dims(image, 0)

            generated = model.net(image, training=False)
            generated = tf.cast(generated, tf.uint8)

            # Remove batch dimension
            generated = tf.squeeze(generated, [0])

            # Restore model variables.
            saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V1)
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            # Use absolute path
            model_file = os.path.abspath(model_file)
            saver.restore(sess, model_file)

            # Make sure 'generated' directory exists.
            generated_file = image_result
            if os.path.exists('generated') is False:
                os.makedirs('generated')

            # Generate and write image data to file.
            with open(generated_file, 'wb') as img:
                start_time = time.time()
                img.write(sess.run(tf.image.encode_jpeg(generated)))
                end_time = time.time()
                tf.logging.info('Elapsed time: %fs' % (end_time - start_time))

                tf.logging.info('Done. Please check %s.' % generated_file)

def change():
    print ("ok")
    
if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

    #change_style()