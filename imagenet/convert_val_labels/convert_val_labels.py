
import numpy as np

label_counter = 0
validation_labels = []

#############################################################

f = open("train_labels.txt", 'r')
lines = f.readlines()
label_counter = 0
order_labels = {}
for line in lines:
    line = line.split()
    order_labels[line[0]] = label_counter
    label_counter += 1
f.close()

f = open("train_labels1.txt", 'r')
lines = f.readlines()
label_to_folder = {}
for line in lines:
    line = line.split()
    label_to_folder[int(line[0])] = line[1]
f.close()

f = open('ILSVRC2012_validation_ground_truth.txt')
lines = f.readlines()
for ii in range(len(lines)):
    true_label = int(lines[ii])
    folder = label_to_folder[true_label]
    order_label = order_labels[folder]
    validation_labels.append(order_label)
    # print (true_label, folder, order_label)
    
f.close()

np.savetxt("validation_labels.txt", validation_labels, fmt='%i', delimiter='\n')
