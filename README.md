# Face-Recognition

To use this ->

first go to train_faces and run FacesFromCamera.py and capture face from different angles (min. 35-40 images)
After running- press N for new faces, press S for capturing, and press Q for Quiting.

Then run facerec_train.py giving argument, the path to data_faces_from_camera (which generated after running FacesFromCamera.py) for face images dataset and give path where encodings.pickle file you want to store.
Example -> > python facerec_train.py -i data_faces_from_camera -e encodings.pickle

After this run the face_test.py for real time face recognition (giving path of the encodings.pickle)
Example -> > python face_test.py -e train_faces/encodings.pickle
