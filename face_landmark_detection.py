#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This example program shows how to find frontal human faces in an image and
#   estimate their pose.  The pose takes the form of 68 landmarks.  These are
#   points on the face such as the corners of the mouth, along the eyebrows, on
#   the eyes, and so forth.
#
#   The face detector we use is made using the classic Histogram of Oriented
#   Gradients (HOG) feature combined with a linear classifier, an image pyramid,
#   and sliding window detection scheme.  The pose estimator was created by
#   using dlib's implementation of the paper:
#      One Millisecond Face Alignment with an Ensemble of Regression Trees by
#      Vahid Kazemi and Josephine Sullivan, CVPR 2014
#   and was trained on the iBUG 300-W face landmark dataset (see
#   https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/):  
#      C. Sagonas, E. Antonakos, G, Tzimiropoulos, S. Zafeiriou, M. Pantic. 
#      300 faces In-the-wild challenge: Database and results. 
#      Image and Vision Computing (IMAVIS), Special Issue on Facial Landmark Localisation "In-The-Wild". 2016.
#   You can get the trained model file from:
#   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2.
#   Note that the license for the iBUG 300-W dataset excludes commercial use.
#   So you should contact Imperial College London to find out if it's OK for
#   you to use this model file in a commercial product.
#
#
#   Also, note that you can train your own models using dlib's machine learning
#   tools. See train_shape_predictor.py to see an example.
#
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake installed.  On Ubuntu, this can be done easily by running the
#   command:
#       sudo apt-get install cmake
#
#   Also note that this example requires Numpy which can be installed
#   via the command:
#       pip install numpy

import sys
import os
import dlib
import glob
import numpy as np
import cv2


def main():

    if len(sys.argv) != 3:
        print(
            "Give the path to the trained shape predictor model as the first "
            "argument and then the directory containing the facial images.\n"
            "For example, if you are in the python_examples folder then "
            "execute this program by running:\n"
            "    ./face_landmark_detection.py shape_predictor_68_face_landmarks.dat ../examples/faces\n"
            "You can download a trained facial shape predictor from:\n"
            "    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        exit()

    predictor_path = sys.argv[1]
    faces_folder_path = sys.argv[2]

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    # win = dlib.image_window()

    for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
        print("Processing file: {}".format(f))
        img = dlib.load_rgb_image(f)
        fo = open(f + "_pts.txt", "a")
        # win.clear_overlay()
        # win.set_image(img)

        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        dets = detector(img, 1)
        # print("Number of faces detected: {}".format(len(dets)))
        img = cv2.imread(f)
        h, w = img.shape[0:2]
        # print(h)
        # print(w)


        h_pts = np.zeros(6)
        w_pts = np.zeros(6)

        h_inc = h/5
        w_inc = w/5

        for i in range(1, 5):
            h_pts[i] = h_inc * i
            w_pts[i] = w_inc * i

        h_pts[5] = h - 1
        w_pts[5] = w - 1
        h_pts = h_pts.astype(np.int)
        w_pts = w_pts.astype(np.int)

        # print(h_pts)
        # print(w_pts)

        for i in range(6):
            for j in range (6):
                x = w_pts[i]
                y = h_pts[j]
                fo.write(str(x) + " " + str(y) + "\n")


        #
        # fo.write("0 0")
        # fo.write()

        for k, d in enumerate(dets):
            # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            #     k, d.left(), d.top(), d.right(), d.bottom()))
            # Get the landmarks/parts for the face in box d.
            shape = predictor(img, d)
            # print("Part 0: {}, Part 1: {} ...".format(shape.part(0), shape.part(1)))
            for z in range(68):
                coord = str(shape.part(z))
                coord = coord.replace("(", "").replace(",", "").replace(")", "")
                # coord = coord.replace(",", "")
                # coord = coord.replace(",", "")
                x, y = coord.split()
                coord = str(x) + " " + str(y) + "\n"
                # print(coord)
                fo.write(str(coord))
            # Draw the face landmarks on the screen.
            # win.add_overlay(shape)

        # win.add_overlay(dets)
        fo.close()
        # dlib.hit_enter_to_continue()

if __name__ == '__main__':
    main()