#!/usr/bin/python3
import matplotlib.pyplot as plt
import pickle
import sys

from Visualize.plotter_3d import plot_clusters


if __name__ == "__main__":
   if len(sys.argv) != 2:
      print("Usage:\n\t.py <fig_pickle_path>")
      sys.exit(1)

   fig_data_path = sys.argv[1]
   if ".pickle" not in fig_data_path.lower():
      print(f"'{fig_data_path}' is not a .pickle file")
      sys.exit(1)

   fig_data = pickle.load(open(fig_data_path, 'rb'))
   plot_clusters(fig_data["clusters"], fig_data["centers"], "", fig_data["key_points"])

   plt.show()

