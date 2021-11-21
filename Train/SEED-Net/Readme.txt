First, using TF_data_train.py to generate tf.record format data.

After the  tf.record format data is generated, run the cycle_gan_train.py and select a path to save the network model.

Finally, You can get the test results by feeding the test data and the saved model to cycle_gan_test.py.


And if you want to batch generate "the generated experiment data"

First, using TF_data_test.py to generate tf.record format data.

After the  tf.record format data is generated, run the cycle_gan_test_save.py and select a path to save the generated data.

