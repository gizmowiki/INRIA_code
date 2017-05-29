import os
import sys

filepath = sys.argv[1]

true_positive = 0
false_positive = 0
false_negative = 0

for lines in open(filepath, 'rb'):
    lines = lines.strip()
    lines = lines.split(",")
    true_positive += int(lines[1])
    false_positive += int(lines[2])
    false_negative += int(lines[3])

precision = true_positive/float((true_positive+false_positive))
recall = true_positive/float((true_positive+false_negative))
accuracy = true_positive/float((true_positive+false_positive+false_negative))
print ("Precision %.3f Recall %.3f Accuracy %.3f " % (precision, recall, accuracy))
