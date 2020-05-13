#!/usr/bin/python3
import matplotlib.pyplot as plt
import pickle
import sys
import bfr


if __name__ == "__main__":
   if len(sys.argv) != 2:
      print("Usage:\n\t.py <fig_pickle_path>")
      sys.exit(1)

   fig_data_path = sys.argv[1]
   if ".pickle" not in fig_data_path.lower():
      print(f"'{fig_data_path}' is not a .pickle file")
      sys.exit(1)

   fig_data = pickle.load(open(fig_data_path, 'rb'))
   model = fig_data["bfr_model"]
   points = fig_data["points"]
   bfr_plot = bfr.plot.BfrPlot(model, points)
   bfr_plot.show()

   sse = model.error(points)
   print(f"sse: {sse: 0.4f}")

