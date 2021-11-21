1. SEED-Net of Style translate
Open the SEED-Net folder, then double-click the “seed_net_test.py” file and configure the environment for the project as described above. Modify the load data and the path to the network model in the “seed_net_test.py” file. The data path is on lines 12 and 17 in the program. The network model data is on line 48.
The data is in the “phantom result” folder inside the “SEED-Net” folder, and the model is in the “model” folder. 

2.QOAT-Net
The steps are like the above. First, open the QOAT-Net folder, and then double-click the “QOAT_Net_test.py” file. Also modify the path of the data and model, on line 10 and line 399, respectively. The network model and data are in the QOAT-Net folder.
If you want to see other results, the example is as follows:

test_input_folder="QOAT-Net\\mouse liver\\input p0.mat" 
saver.restore(sess, " QOAT-Net\\mouse liver\\model\\model.ckpt")
