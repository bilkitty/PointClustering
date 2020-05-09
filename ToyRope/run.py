#!/usr/bin/python3
import spline
import sys

PCD_FILEPATH = "/home/bilkit/PycharmProjects/ToyRope/test.pcd"

def main():
    if len(sys.argv) != 3:
        print("USAGE:\n  run.py <type> <# points>")
        return 1
    selection = int(sys.argv[1])
    size = int(sys.argv[2])
    if int(selection) == 0:
        spline.display_spline(size)
    elif int(selection) == 1:
        spline.display_spline_points(size)
    elif int(selection) == 2:
        spline.display_spline_points(size, PCD_FILEPATH)
    else:
        print("Unknown")
        return 1


if __name__ == "__main__":
    main()
    print("done :)")