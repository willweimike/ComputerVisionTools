import sys
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                           QHBoxLayout, QWidget, QGroupBox, QFileDialog, QLabel, 
                           QTextEdit, QComboBox, QMessageBox)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import numpy as np
import glob


# class ImageWindow(QMainWindow):
#     def __init__(self, title, image_path, x_pos, size=(400, 300)):
#         super().__init__()
#         self.setWindowTitle(title)
#         self.setGeometry(x_pos, 100, size[0], size[1])
        
#         # Create label to hold the image
#         self.image_label = QLabel(self)
#         self.image_label.setAlignment(Qt.AlignCenter)
        
#         # Load and scale the image
#         pixmap = QPixmap(image_path)
#         scaled_pixmap = pixmap.scaled(size[0]-20, size[1]-20, 
#                                     Qt.KeepAspectRatio, 
#                                     Qt.SmoothTransformation)
        
#         self.image_label.setPixmap(scaled_pixmap)
#         self.image_label.resize(size[0], size[1])
#         self.image_label.setAlignment(Qt.AlignCenter)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Computer Vision Tool")
        self.setGeometry(100, 100, 800, 600)
        self.image_paths = []
        self.left_image_path = ""
        self.right_image_path = ""
        self.corners_result = []
        self.image_1_path = ""
        self.image_2_path = ""
        self.dropdown = QComboBox()
        self.num_imgs = 0
        self.text_edit = QTextEdit()
        self.ret = []
        self.ins = []
        self.dist = []
        self.rvec = []
        self.tvec = []
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # Create blocks
        self.create_load_image_block(main_layout)
        self.create_calibration_block(main_layout)
        self.create_augment_reality_block(main_layout)
        self.create_stereo_disparity_block(main_layout)
        self.create_sift_block(main_layout)

    def create_load_image_block(self, parent_layout):
        group_box = QGroupBox("Load Image")
        layout = QVBoxLayout()
        
        btn_load_folder = QPushButton("Load Folder")
        btn_load_image_l = QPushButton("Load Image_L")
        btn_load_image_r = QPushButton("Load Image_R")
        
        # Connect buttons to their functions
        btn_load_folder.clicked.connect(self.load_folder_image)
        btn_load_image_l.clicked.connect(self.load_image_left)
        btn_load_image_r.clicked.connect(self.load_image_right)
        
        layout.addWidget(btn_load_folder)
        layout.addWidget(btn_load_image_l)
        layout.addWidget(btn_load_image_r)
        
        group_box.setLayout(layout)
        parent_layout.addWidget(group_box)

    def create_calibration_block(self, parent_layout):
        group_box = QGroupBox("1. Calibration")
        layout = QVBoxLayout()
        
        buttons = ["1.1 Find corners", "1.2 Find intrinsic", "1.3 Find extrinsic", 
                  "1.4 Find distortion", "1.5 Show result"]
        
        for text in buttons:
            btn = QPushButton(text)
            if text == "1.1 Find corners":
                btn.clicked.connect(lambda: self.corner_detect(self.image_paths))
            if text == "1.2 Find intrinsic":
                btn.clicked.connect(lambda: self.find_intrinsic(self.image_paths))
            if text == "1.3 Find extrinsic":
                layout.addWidget(self.dropdown)
                btn.clicked.connect(lambda: self.find_extrinsic(self.image_paths))
            if text == "1.4 Find distortion":
                btn.clicked.connect(lambda: self.find_distortion(self.image_paths))
            if text == "1.5 Show result":
                btn.clicked.connect(lambda: self.show_result_window(self.image_paths))
            layout.addWidget(btn)
        
        group_box.setLayout(layout)
        parent_layout.addWidget(group_box)

    def create_augment_reality_block(self, parent_layout):
        group_box = QGroupBox("2. Augment Reality")
        box_layout = QVBoxLayout()
        
        self.text_edit.setMaximumSize(200, 50)
        box_layout.addWidget(self.text_edit)


        btn_layout = QVBoxLayout()
        buttons = ["2.1 Show words on board", "2.2 Show words vertical"]
        
        for text in buttons:
            btn = QPushButton(text)
            if text == "2.1 Show words on board":
                btn.clicked.connect(lambda: self.show_words_on_board(self.image_paths))
            if text == "2.2 Show words vertical":
                btn.clicked.connect(lambda: self.show_words_vertical(self.image_paths))
            btn_layout.addWidget(btn)
        self.text_edit.textChanged.connect(self.get_text)

        box_layout.addLayout(btn_layout)
        group_box.setLayout(box_layout)
        parent_layout.addWidget(group_box)
    
    def get_text(self):
        content = self.text_edit.toPlainText()
        content = content.upper()
        return content

    def create_stereo_disparity_block(self, parent_layout):
        group_box = QGroupBox("3. Stereo disparity map")
        layout = QVBoxLayout()
        
        btn_disparity = QPushButton("3.1 stereo disparity map")
        btn_disparity.clicked.connect(lambda: self.stereo_disparity_map(self.left_image_path, self.right_image_path))
        layout.addWidget(btn_disparity)
        
        group_box.setLayout(layout)
        parent_layout.addWidget(group_box)

    def create_sift_block(self, parent_layout):
        group_box = QGroupBox("4. SIFT")
        layout = QVBoxLayout()
        
        buttons = ["Load Image 1", "Load Image 2", "4.1 Keypoints", "4.2 Matched Keypoints"]
        
        for text in buttons:
            btn = QPushButton(text)
            if text == "Load Image 1":
                btn.clicked.connect(lambda: self.load_image_1())
            if text == "Load Image 2":
                btn.clicked.connect(lambda: self.load_image_2())
            if text == "4.1 Keypoints":
                btn.clicked.connect(lambda: self.find_keypoints(self.image_1_path))
            if text == "4.2 Matched Keypoints":
                btn.clicked.connect(lambda: self.match_keypoints(self.image_1_path, self.image_2_path))
            layout.addWidget(btn)
        
        group_box.setLayout(layout)
        parent_layout.addWidget(group_box)

    def load_image_left(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Left Image", "", 
                                                 "Image Files (*.png *.jpg *.bmp)")
        if file_path:
            self.left_image_path = file_path
            print(f"Left image path: {file_path}")
           
        return self.left_image_path

    def load_image_right(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Right Image", "", 
                                                 "Image Files (*.png *.jpg *.bmp)")
        if file_path:
            self.right_image_path = file_path
            print(f"Right image path: {file_path}")
            
        return self.right_image_path

    def load_folder_image(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            # Get all image files in the folder
            
            image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
            self.image_paths = []
            for ext in image_extensions:
                self.image_paths.extend(glob.glob(f"{folder_path}/{ext}"))
            
            self.num_imgs = len(self.image_paths)
            self.dropdown.addItems([str(i) for i in range(1, self.num_imgs+1)])
            
            print(f"Loaded {len(self.image_paths)} images from folder")
            return self.image_paths, self.num_imgs

    def corner_detect(self, image_paths):
        self.corners_result = []
        if isinstance(image_paths, list) and image_paths != []:
            for path in image_paths:
                img = cv2.imread(path)
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                b_height, b_width = 8, 11

                ret, corners = cv2.findChessboardCorners(gray_img, (b_width, b_height))
                corners = corners.astype(np.float32)

                winSize = (5, 5)
                zeroZone = (-1, -1)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corner_point = cv2.cornerSubPix(gray_img, corners, winSize, zeroZone, criteria)
                self.corners_result.append(corner_point)

                img_with_corners = img.copy()
                cv2.drawChessboardCorners(img_with_corners, (b_width, b_height), corner_point, ret)

                cv2.imshow('Chessboard Corners', img_with_corners)
                cv2.waitKey(500)
                cv2.destroyAllWindows()

            return self.corners_result
        else:
            print("No image loaded.")

    
    def find_intrinsic(self, image_paths):
        if isinstance(image_paths, list) and image_paths != []:
            for path in image_paths:
                img = cv2.imread(path)
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                b_height, b_width = 8, 11

                ret, corners = cv2.findChessboardCorners(gray_img, (b_width, b_height))
                corners = corners.astype(np.float32)

                winSize = (5, 5)
                zeroZone = (-1, -1)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corner_point = cv2.cornerSubPix(gray_img, corners, winSize, zeroZone, criteria)
            
                b_height, b_width = 8, 11
                objp = np.zeros((b_width*b_height, 3), np.float32)
                objp[:, :2] = np.mgrid[0:b_width, 0:b_height].T.reshape(-1, 2)

                objectPoints = [objp]
                imgPoints = [corner_point]
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                
                ret, ins, dist, rvec, tvec = cv2.calibrateCamera(objectPoints=objectPoints, imagePoints=imgPoints, imageSize=gray_img.shape[::-1], cameraMatrix=None, distCoeffs=None, criteria=criteria)
                self.ret.append(ret)
                self.ins.append(ins)
                self.dist.append(dist)
                self.rvec.append(rvec)
                self.tvec.append(tvec)
                print(f"Intrinsic Matrix:\n{ins}")
                QMessageBox.information(self, "Intrinsic Matrix", f"Intrinsic Matrix:\n{ins}")

        else:
            print("No image loaded.")
    

    def find_extrinsic(self, image_paths):
        if isinstance(image_paths, list) and image_paths != []:
            for path in image_paths:
                img = cv2.imread(path)
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                b_height, b_width = 8, 11

                ret, corners = cv2.findChessboardCorners(gray_img, (b_width, b_height))
                corners = corners.astype(np.float32)

                winSize = (5, 5)
                zeroZone = (-1, -1)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corner_point = cv2.cornerSubPix(gray_img, corners, winSize, zeroZone, criteria)
            
                b_height, b_width = 8, 11
                objp = np.zeros((b_width*b_height, 3), np.float32)
                objp[:, :2] = np.mgrid[0:b_width, 0:b_height].T.reshape(-1, 2)

                objectPoints = [objp]
                imgPoints = [corner_point]
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                
                ret, ins, dist, rvec, tvec = cv2.calibrateCamera(objectPoints=objectPoints, imagePoints=imgPoints, imageSize=gray_img.shape[::-1], cameraMatrix=None, distCoeffs=None, criteria=criteria)
                self.ret.append(ret)
                self.ins.append(ins)
                self.dist.append(dist)
                self.rvec.append(rvec)
                self.tvec.append(tvec)
            
            current_img = self.dropdown.currentText()
            current_img_index = int(current_img) - 1

            current_img_rvec = self.rvec[current_img_index]
            rotation_matrix, _ = cv2.Rodrigues(current_img_rvec[0])

            current_img_tvec = self.tvec[current_img_index]
            translation_matrix = current_img_tvec[0]

            extrinsic_matrix = np.hstack((rotation_matrix , translation_matrix))
            print(f"Extrinsic Matrix:\n{extrinsic_matrix}")
            QMessageBox.information(self, "Extrinsic Matrix", f"Extrinsic Matrix:\n{extrinsic_matrix}")
        else:
            print("No image loaded.")

    def find_distortion(self, image_paths):
        if isinstance(image_paths, list) and image_paths != []:
            self.corners_result = []
            self.ret = []
            self.ins = []
            self.dist = []
            self.rvec = []
            self.tvec = []

            for path in image_paths:
                img = cv2.imread(path)
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                b_height, b_width = 8, 11

                ret, corners = cv2.findChessboardCorners(gray_img, (b_width, b_height))
                corners = corners.astype(np.float32)

                winSize = (5, 5)
                zeroZone = (-1, -1)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corner_point = cv2.cornerSubPix(gray_img, corners, winSize, zeroZone, criteria)
            
                b_height, b_width = 8, 11
                objp = np.zeros((b_width*b_height, 3), np.float32)
                objp[:, :2] = np.mgrid[0:b_width, 0:b_height].T.reshape(-1, 2)

                objectPoints = [objp]
                imgPoints = [corner_point]
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                
                ret, ins, dist, rvec, tvec = cv2.calibrateCamera(objectPoints=objectPoints, imagePoints=imgPoints, imageSize=gray_img.shape[::-1], cameraMatrix=None, distCoeffs=None, criteria=criteria)
                print(f"Distortion Matrix:\n{dist}")
                QMessageBox.information(self, "Distortion Matrix", f"Distortion Matrix:\n{dist}")

        else:
            print("No image loaded.")
    
    def show_result_window(self, image_paths):
        if isinstance(image_paths, list) and image_paths != []:
            for path in image_paths:
                img = cv2.imread(path)
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                b_height, b_width = 8, 11

                ret, corners = cv2.findChessboardCorners(gray_img, (b_width, b_height))
                corners = corners.astype(np.float32)

                winSize = (5, 5)
                zeroZone = (-1, -1)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corner_point = cv2.cornerSubPix(gray_img, corners, winSize, zeroZone, criteria)
            
                b_height, b_width = 8, 11
                objp = np.zeros((b_width*b_height, 3), np.float32)
                objp[:, :2] = np.mgrid[0:b_width, 0:b_height].T.reshape(-1, 2)

                objectPoints = [objp]
                imgPoints = [corner_point]
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                
                ret, ins, dist, rvec, tvec = cv2.calibrateCamera(objectPoints=objectPoints, imagePoints=imgPoints, imageSize=gray_img.shape[::-1], cameraMatrix=None, distCoeffs=None, criteria=criteria)
                        
                result_img = cv2.undistort(gray_img, ins, dist)
                # cv2.imwrite("result.jpg", result_img)

                # window1 = ImageWindow("Distorted Image", path, x_pos=100)
                # window1.show()
                # window2 = ImageWindow("Undistorted Image", "result.jpg", x_pos=550)
                # window2.show()

                cv2.imshow("Original", img)
                cv2.waitKey(0)
                cv2.imshow("Result", result_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
        else:
            print("No image loaded.")
    
    def show_words_on_board(self, image_paths):
        if isinstance(image_paths, list) and image_paths != []:
            text_input = self.get_text()
            for path in image_paths:
                img = cv2.imread(path)
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                b_height, b_width = 8, 11

                ret, corners = cv2.findChessboardCorners(gray_img, (b_width, b_height))
                print(corners)
                print('\n')
                corners = corners.astype(np.float32)

                winSize = (5, 5)
                zeroZone = (-1, -1)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corner_point = cv2.cornerSubPix(gray_img, corners, winSize, zeroZone, criteria)
            
                b_height, b_width = 8, 11
                objp = np.zeros((b_width*b_height, 3), np.float32)
                objp[:, :2] = np.mgrid[0:b_width, 0:b_height].T.reshape(-1, 2)

                objectPoints = [objp]
                imgPoints = [corner_point]
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

                objectPoints = [np.array(objp, dtype=np.float32) for objp in objectPoints]
                imgPoints = [np.array(corner_point, dtype=np.float32) for corner_point in imgPoints]
                            
                ret, ins, dist, rvec, tvec = cv2.calibrateCamera(objectPoints=objectPoints, imagePoints=imgPoints, imageSize=gray_img.shape[::-1], cameraMatrix=None, distCoeffs=None, criteria=criteria)

                fs = cv2.FileStorage('/Users/liuchengwei/cvdl/Dataset_CvDl_Hw1/Q2_Image/Q2_db/alphabet_db_onboard.txt', cv2.FILE_STORAGE_READ)

                ins = ins.astype(np.float32)
                dist = dist.astype(np.float32)
                new_rvec = rvec[0].astype(np.float32)
                new_tvec = tvec[0].astype(np.float32)
                
                corner_positions = [
                    (8, 5, 0),
                    (6, 5, 0),
                    (4, 5, 0),
                    (8, 2, 0), 
                    (6, 2, 0),
                    (4, 2, 0)
                ]
                
                char_spacing = 4
                
                for word_idx, word in enumerate(text_input.split()):
                    if word_idx >= len(corner_positions):
                        break
                    
                    base_position = corner_positions[word_idx]
                    
                    for char_idx, char in enumerate(word):
                        if char_idx < 3:
                            char_position = (
                                base_position[0] - char_idx * char_spacing,
                                base_position[1],
                                base_position[2]
                            )
                        else:
                            char_position = (
                            base_position[0] - (char_idx-3) * char_spacing,
                            base_position[1] - 3,
                            base_position[2]
                        )
                        
                        charPoints = fs.getNode(char).mat()
                        charPoints = charPoints.tolist()
                        
                        for i in range(len(charPoints)):
                            inner_charPoints_list = charPoints[i]
                            inner_charPoints = np.array(inner_charPoints_list, dtype=np.float32)
                            
                            translated_points = inner_charPoints.copy()
                            translated_points[:, 0] += char_position[0]
                            translated_points[:, 1] += char_position[1]
                            translated_points[:, 2] += char_position[2]
                            
                            newCharPoints, jacobian = cv2.projectPoints(translated_points, new_rvec, new_tvec, ins, dist)
                            newCharPoints_2d = newCharPoints.reshape(-1, 2).astype(np.int32)
                            newCharPoints_2d_list = newCharPoints_2d.tolist()
                            
                            pt1 = tuple(newCharPoints_2d_list[0])
                            pt2 = tuple(newCharPoints_2d_list[1])
                            cv2.line(img, pt1, pt2, (0, 0, 255), 2)

                cv2.imshow("Result", img)
                cv2.waitKey(1000)
                cv2.destroyAllWindows()                              
        else:
            print("No image loaded.")

    def show_words_vertical(self, image_paths):
        if isinstance(image_paths, list) and image_paths != []:
            text_input = self.get_text()
            for path in image_paths:
                img = cv2.imread(path)
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                b_height, b_width = 8, 11

                ret, corners = cv2.findChessboardCorners(gray_img, (b_width, b_height))
                corners = corners.astype(np.float32)

                winSize = (5, 5)
                zeroZone = (-1, -1)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corner_point = cv2.cornerSubPix(gray_img, corners, winSize, zeroZone, criteria)
            
                b_height, b_width = 8, 11
                objp = np.zeros((b_width*b_height, 3), np.float32)
                objp[:, :2] = np.mgrid[0:b_width, 0:b_height].T.reshape(-1, 2)

                objectPoints = [objp]
                imgPoints = [corner_point]
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

                objectPoints = [np.array(objp, dtype=np.float32) for objp in objectPoints]
                imgPoints = [np.array(corner_point, dtype=np.float32) for corner_point in imgPoints]
                            
                ret, ins, dist, rvec, tvec = cv2.calibrateCamera(objectPoints=objectPoints, imagePoints=imgPoints, imageSize=gray_img.shape[::-1], cameraMatrix=None, distCoeffs=None, criteria=criteria)

                fs = cv2.FileStorage('/Users/liuchengwei/cvdl/Dataset_CvDl_Hw1/Q2_Image/Q2_db/alphabet_db_vertical.txt', cv2.FILE_STORAGE_READ)

                ins = ins.astype(np.float32)
                dist = dist.astype(np.float32)
                new_rvec = rvec[0].astype(np.float32)
                new_tvec = tvec[0].astype(np.float32)
                
                corner_positions = [
                    (8, 5, 0),
                    (6, 5, 0),
                    (4, 5, 0),
                    (8, 2, 0),
                    (6, 2, 0),
                    (4, 2, 0)
                ]
                
                char_spacing = 4
                
                for word_idx, word in enumerate(text_input.split()):
                    if word_idx >= len(corner_positions):
                        break
                    
                    base_position = corner_positions[word_idx]
                    
                    for char_idx, char in enumerate(word):
                        if char_idx < 3:
                            char_position = (
                                base_position[0] - char_idx * char_spacing,
                                base_position[1],
                                base_position[2]
                            )
                        else:
                            char_position = (
                            base_position[0] - (char_idx-3) * char_spacing,
                            base_position[1] - 3,
                            base_position[2]
                        )
                        
                        charPoints = fs.getNode(char).mat()
                        charPoints = charPoints.tolist()
                        
                        for i in range(len(charPoints)):
                            inner_charPoints_list = charPoints[i]
                            inner_charPoints = np.array(inner_charPoints_list, dtype=np.float32)
                            
                            translated_points = inner_charPoints.copy()
                            translated_points[:, 0] += char_position[0]
                            translated_points[:, 1] += char_position[1]
                            translated_points[:, 2] += char_position[2]
                            
                            newCharPoints, jacobian = cv2.projectPoints(translated_points, new_rvec, new_tvec, ins, dist)
                            newCharPoints_2d = newCharPoints.reshape(-1, 2).astype(np.int32)
                            newCharPoints_2d_list = newCharPoints_2d.tolist()
                            
                            pt1 = tuple(newCharPoints_2d_list[0])
                            pt2 = tuple(newCharPoints_2d_list[1])
                            cv2.line(img, pt1, pt2, (0, 0, 255), 2)

                cv2.imshow("Result", img)
                cv2.waitKey(1000)
                cv2.destroyAllWindows()
        else:
            print("No image loaded.")

    def stereo_disparity_map(self, left_image_path, right_image_path):
        if isinstance(left_image_path, str) and isinstance(right_image_path, str) and left_image_path != "" and right_image_path != "":
            imgL = cv2.imread(left_image_path)
            imgR = cv2.imread(right_image_path)
            gray_img1 = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
            gray_img2 = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
            
            stereo = cv2.StereoBM.create(numDisparities=432, blockSize=25)
            disparity = stereo.compute(gray_img1, gray_img2).astype(np.float32) / 16.0

            normalized_disparity = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            disparity_colored_map = cv2.applyColorMap(normalized_disparity, cv2.COLORMAP_JET)
            disparity_gray_map = cv2.cvtColor(disparity_colored_map, cv2.COLOR_BGR2GRAY)
            # # Read and display image
            # if imgL is not None:
            #     l_img = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
            #     lh, lw, lch = l_img.shape
            #     l_bytes_per_line = lch * lw
                
            #     # Convert to QImage and display
            #     l_qt_img = QImage(l_img.data, lw, lh, l_bytes_per_line, QImage.Format_RGB888)
                
            #     # Create new window to display image
            #     self.l_image_window = QWidget()
            #     self.l_image_window.setWindowTitle("ImgL")
            #     l_layout = QVBoxLayout()
            #     l_label = QLabel()
            #     l_label.setPixmap(QPixmap.fromImage(l_qt_img))
            #     l_layout.addWidget(l_label)
            #     self.l_image_window.setLayout(l_layout)
            #     self.l_image_window.show()
            
            # # Read and display image
            # if imgR is not None:
            #     r_img = cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB)
            #     rh, rw, rch = r_img.shape
            #     r_bytes_per_line = rch * rw
                
            #     # Convert to QImage and display
            #     r_qt_img = QImage(r_img.data, rw, rh, r_bytes_per_line, QImage.Format_RGB888)
                
            #     # Create new window to display image
            #     self.r_image_window = QWidget()
            #     self.r_image_window.setWindowTitle("ImgR")
            #     r_layout = QVBoxLayout()
            #     r_label = QLabel()
            #     r_label.setPixmap(QPixmap.fromImage(r_qt_img))
            #     r_layout.addWidget(r_label)
            #     self.r_image_window.setLayout(r_layout)
            #     self.r_image_window.show()

            # if disparity_gray_map is not None:
            #     h, w = disparity_gray_map.shape
            #     bytes_per_line = w
                
            #     # Convert to QImage and display
            #     qt_img = QImage(disparity_gray_map.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
                
            #     # Create new window to display image
            #     self.image_window = QWidget()
            #     self.image_window.setWindowTitle("disparity map")
            #     layout = QVBoxLayout()
            #     label = QLabel()
            #     label.setPixmap(QPixmap.fromImage(qt_img))
            #     layout.addWidget(label)
            #     self.image_window.setLayout(layout)
            #     self.image_window.show()

            cv2.imshow("ImgL", imgL)
            cv2.waitKey(0)
            cv2.imshow("ImgR", imgR)
            cv2.waitKey(0)
            cv2.imshow("disparity map", disparity_gray_map)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No left or right image loaded.")

    def load_image_1(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image 1", "", 
                                                 "Image Files (*.png *.jpg *.bmp)")
        if file_path:
            self.image_1_path = file_path
            print(f"Image 1 path: {file_path}")

        return self.image_1_path

    def load_image_2(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image 2", "", 
                                                 "Image Files (*.png *.jpg *.bmp)")
        if file_path:
            self.image_2_path = file_path
            print(f"Image 2 path: {file_path}")

        return self.image_2_path
    
    def find_keypoints(self, image_1_path):
        if isinstance(image_1_path, str) and image_1_path != "":
            img1 = cv2.imread(image_1_path)
            gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            sift = cv2.SIFT.create()
            keypoints, descriptors = sift.detectAndCompute(gray, None)

            draw_img = cv2.drawKeypoints(gray, keypoints, None, color=(0,255,0))
            # # Read and display image
            # if draw_img is not None:
            #     g_img = cv2.cvtColor(draw_img, cv2.COLOR_BGR2GRAY)
            #     h, w = g_img.shape
            #     bytes_per_line = w
                
            #     # Convert to QImage and display
            #     qt_img = QImage(g_img, w, h, bytes_per_line, QImage.Format_Grayscale8)
                
            #     # Create new window to display image
            #     self.image_window = QWidget()
            #     self.image_window.setWindowTitle("Keypoints")
            #     layout = QVBoxLayout()
            #     label = QLabel()
            #     label.setPixmap(QPixmap.fromImage(qt_img))
            #     layout.addWidget(label)
            #     self.image_window.setLayout(layout)
            #     self.image_window.show()

            cv2.imshow("Keypoints", draw_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No image loaded.")
        
    def match_keypoints(self, image_1_path, image_2_path):
        if isinstance(image_1_path, str) and isinstance(image_2_path, str) and image_1_path != ""  and image_2_path != "":
            img1 = cv2.imread(image_1_path)
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            sift1 = cv2.SIFT.create()
            keypoints1, descriptors1 = sift1.detectAndCompute(gray1, None)

            img2 = cv2.imread(image_2_path)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            sift2 = cv2.SIFT.create()
            keypoints2, descriptors2 = sift2.detectAndCompute(gray2, None)

            matches = cv2.BFMatcher().knnMatch(descriptors1, descriptors2, k=2)
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75*n.distance:
                    good_matches.append(m)
            
            img_matches = cv2.drawMatches(
                gray1, keypoints1,
                gray2, keypoints2,
                good_matches, None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            
            cv2.imshow("Matched Keypoints", img_matches)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No image loaded.")

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
