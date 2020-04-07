from ROOT import TCanvas, TH1F, TPad, TFile, TLegend, TPaveLabel, TLatex, TPaveText, gStyle
from ROOT import gROOT
import numpy as np
import h5py


test_file = h5py.File("/Volumes/MacOS/Research/Data/high-level/test_no_pile_5000000.h5", "r")
train_file = h5py.File("/Volumes/MacOS/Research/Data/high-level/train_no_pile_10000000.h5", "r")
f=TFile("High_Level_Variables.root","recreate")

X_train = train_file['features']
X_test = test_file['features']

y_train = train_file['targets']
y_test = test_file['targets']

N_train = len(X_train)
#N_test = len(X_test)
#N_train = N_test = 60000

train_vars=np.array(X_train[0:N_train])
train_labels=np.array(y_train[0:N_train])

# test_vars=np.array(X_test[0:N_test])
# test_labels=np.array(y_test[0:N_test])

print("Read Finished")


signal_index = np.where(train_labels == 1.)
background_index = np.where(train_labels == 0)

variables_signal = ['mass_signal', 'C_21_signal', 'C_22_signal', 'D_21_signal', 'D_22_signal', 'tau21_signal']
variables_background = ['mass_background', 'C_21_background', 'C_22_background', 'D_21_background', 'D_22_background', 'tau21_background']

variables_range = [200, 0.5, 0.2, 5., 5., 1.]
variables_name = ['Trimmed Mass', 'C_21', 'C_22', 'D_21', 'D_22', 'tau21']


print("Select Finished")

for x in range(0,6):
	h1f = TH1F( variables_signal[x], variables_name[x] , 100, 0, variables_range[x] )
	h2f = TH1F( variables_background[x], variables_name[x] , 100, 0, variables_range[x] )
	for i in range(len(signal_index[0])):
		h1f.Fill(train_vars[signal_index[0][i],x])
	for i in range(len(background_index[0])):
		h2f.Fill(train_vars[background_index[0][i],x])
	pass

	h1f.Write()
	h2f.Write()

	del h1f, h2f
	print(x)

f.Close()
# variables_signal = ['mass_signal', 'C_21_signal', 'C_22_signal', 'D_21_signal', 'D_22_signal', 'tau21_signal']
# variables_background = ['mass_background', 'C_21_background', 'C_22_background', 'D_21_background', 'D_22_background', 'tau21_background']

# variables_range = [200, 0.5, 0.2, 5., 5., 1.]
# variables_name = ['Trimmed Mass', 'C_21', 'C_22', 'D_21', 'D_22', 'tau21']

# for x in range(0,6):
#     h1f = TH1F( variables_signal[x], variables_name[x] , 100, 0, variables_range[x] )
#     h2f = TH1F( variables_background[x], variables_name[x] , 100, 0, variables_range[x] )
#     for i in range(0, len(files)):        
#         #Num = 100
#         features = files[i]['features']
#         targets = files[i]['targets']
#         Num = len(features)
#         for j in range(0,Num):
#             if targets[j] == 1. :
#                 h1f.Fill(features[j,x])
#                 pass
#             else : 
#                 h2f.Fill(features[j,x])
#             pass
#         del features, targets
#     h1f.Write()
#     h2f.Write()

#     del h1f, h2f
#     print(x)



# files = [test_file, train_file]
# #f=TFile("High_Level_Variables.root","recreate")
# f=TFile("test.root","recreate")

# h1f = TH1F( 'h1f', 'Trimmed Mass ', 100, 0, 200 )
# h2f = TH1F( 'h2f', 'Trimmed Mass ', 100, 0, 200 )

# signal_index = np.where(train_labels == 1.)
# background_index = np.where(train_labels == 0)

# for i in range(len(signal_index[0])):
#     h1f.Fill(train_vars[signal_index[0][i],0])

# for i in range(len(background_index[0])):
#     h2f.Fill(train_vars[background_index[0][i],0])

# h1f.Write()
# h2f.Write()

