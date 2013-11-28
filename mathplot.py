import matplotlib.pyplot as plt

# plt.plot(range(10),range(10))
# plt.show()

#make a new figure
# fig = plt.figure()

# # make a new axis on that figure. Syntax for add_subplot() is
# # number of rows of subplots, number of columns, and the
# # which subplot. So this says one row, one column, first
# # subplot -- the simplest setup you can get.
# # See later examples for more.

# ax = fig.add_subplot(1,1,1)

# # your data here:     
# x = [1,2,3]
# y = [4,6,3]

# # add a bar plot to the axis, ax.
# ax.bar(x,y)

# # after you're all done with plotting commands, show the plot.
# plt.show()

import numpy as np


N = 5
menMeans   = (.20, .35, .30, .35, .27)
womenMeans = (.25, .32, .34, .20, .25)
ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, menMeans,   width, color='r')
p2 = plt.bar(ind+width, womenMeans, width, color='y')

plt.ylabel('Accuracy')
plt.title('AAAC Problem Accuracy')
plt.xticks(ind+width/2., ('G1', 'G2', 'G3', 'G4', 'G5') )
plt.yticks(np.arange(0.0,1.0,20))
plt.legend( (p1[0], p2[0]), ('Baseline', 'Algorithm') )

plt.show()
