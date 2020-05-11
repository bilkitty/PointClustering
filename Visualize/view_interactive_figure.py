#!/usr/bin/python3
import matplotlib.pyplot as plt
import pickle
import sys


if __name__ == "__main__":
   if len(sys.argv) != 2:
      print("Usage:\n\t.py <fig_pickle_path>")
      sys.exit(1)

   fig_path = sys.argv[1]
   if ".pickle" not in fig_path.lower():
      print(f"'{fig_path}' is not a .pickle file")
      sys.exit(1)

   fig = pickle.load(open(fig_path, 'rb'))
   plt.show()

