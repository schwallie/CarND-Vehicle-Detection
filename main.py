"""
(big image -> resize->hog_features->then sliding windows)
"""
import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from scipy.ndimage.measurements import label

import config
from lessons import *


class DetectVehicles(object):
    def __init__(self, color_space='YCrCb', spatial_size=(32, 32),
                 hist_bins=32, orient=9, single_image=True,
                 pix_per_cell=8, cell_per_block=2, hog_channel=0,
                 spatial_feat=True, hist_feat=False, hog_feat=True,
                 fit_model=None, X_scaler=None, scale_img=False):
        self.color_space = color_space
        self.spatial_size = spatial_size
        self.hist_bins = hist_bins
        self.orient = orient
        self.single_image = single_image
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.hog_channel = hog_channel
        self.spatial_feat = spatial_feat
        self.hist_feat = hist_feat
        self.hog_feat = hog_feat
        self.fit_model = fit_model
        self.X_scaler = X_scaler
        self.frame_num = 0
        self.scale_img = scale_img

    def draw_on_image(self, img):
        # cv2.imwrite('video_input_images/final_{0}.png'.format(self.frame_num), img)
        # Uncomment the following line if you extracted training
        # data from .png images (scaled 0 to 1 by mpimg) and the
        # image you are searching is a .jpg (scaled 0 to 255)
        draw_image = np.copy(img)
        if self.scale_img:
            draw_image = draw_image.astype(np.float32) / 255
        # img = img.astype(np.float32) / 255
        windows = self.slide_window(draw_image, x_start_stop=[690, None], y_start_stop=[375, 430],
                                    xy_window=(110, 90), xy_overlap=config.xy_overlap)
        windows += self.slide_window(draw_image, x_start_stop=[760, None], y_start_stop=[375, 560],
                                         xy_window=(110, 90), xy_overlap=config.xy_overlap)
        hot_windows = self.search_windows(draw_image, windows, self.fit_model, self.X_scaler)

        # draw_img = self.draw_boxes(img, hot_windows, color=(0, 0, 200), thick=6)
        heat = np.zeros_like(img[:, :, 0]).astype(np.float)
        heat = self.add_heat(heat, hot_windows)
        heat2 = self.apply_threshold(heat, 7, img)
        # cv2.imwrite('video_output_images/final_{0}.png'.format(self.frame_num), draw_img)
        self.frame_num += 1
        return heat2

    def compute_features(self, feature_image):
        file_features = []
        if self.spatial_feat:
            spatial_features = bin_spatial(feature_image, size=self.spatial_size)
            file_features.append(spatial_features)
        if self.hist_feat:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=self.hist_bins)
            file_features.append(hist_features)
        if self.hog_feat:
            # Call get_hog_features() with vis=False, feature_vec=True
            if self.hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                         self.orient, self.pix_per_cell, self.cell_per_block,
                                                         vis=True, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:, :, self.hog_channel], self.orient,
                                                self.pix_per_cell, self.cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        return file_features

    def apply_threshold(self, heatmap, threshold, img):
        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0
        # Return thresholded map
        labels = label(heatmap)
        print(labels[1])
        for car_number in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
        # cv2.imwrite('heat{0}.png'.format(pd.datetime.now().microsecond), img)
        return img

    def add_heat(self, heatmap, bbox_list):
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        # Return updated heatmap
        return heatmap

    # Define a function to extract features from a list of images
    # Have this function call bin_spatial() and color_hist()
    def extract_features(self, img, single_image=None):
        # Allows override
        if single_image:
            self.single_image = single_image
        # List of images
        if not self.single_image:
            # Create a list to append feature vectors to
            features = []
            # Iterate through the list of images
            for file in img:
                # Read in each one by one
                image = cv2.imread(file)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                feature_image = self.cvt_color(image)
                file_features = self.compute_features(feature_image)
                features.append(np.concatenate(file_features))
            # Return list of feature vectors
            return features
        else:
            # 2) Apply color conversion if other than 'RGB'
            feature_image = self.cvt_color(img)
            img_features = self.compute_features(feature_image)
            # 9) Return concatenated array of features
            return np.concatenate(img_features)

    # Define a function you will pass an image
    # and the list of windows to be searched (output of slide_windows())
    def search_windows(self, img, windows, clf, scaler):
        # 1) Create an empty list to receive positive detection windows
        on_windows = []
        # 2) Iterate over all windows in the list
        for window in windows:
            # 3) Extract the test window from original image
            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
            # 4) Extract features for that window using single_img_features()
            # HACKY
            saved_single = self.single_image
            features = self.extract_features(test_img, True)
            self.single_image = saved_single
            # 5) Scale extracted features to be fed to classifier
            test_features = scaler.transform(np.array(features).reshape(1, -1))
            # 6) Predict using your classifier
            prediction = clf.predict(test_features)
            # 7) If positive (prediction == 1) then save the window
            if prediction == 1:
                on_windows.append(window)
        # 8) Return windows for positive detections
        return on_windows

    def cvt_color(self, image):
        # apply color conversion if other than 'RGB'
        color_space = self.color_space
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB)
        else:
            feature_image = np.copy(image)
        return feature_image

    # Define a function that takes an image,
    # start and stop positions in both x and y,
    # window size (x and y dimensions),
    # and overlap fraction (for both x and y)
    def slide_window(self, img, x_start_stop=[None, None], y_start_stop=[None, None],
                     xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
        # If x and/or y start/stop positions not defined, set to image size
        if x_start_stop[0] is None:
            x_start_stop[0] = 0
        if x_start_stop[1] is None:
            x_start_stop[1] = img.shape[1]
        if y_start_stop[0] is None:
            y_start_stop[0] = 0
        if y_start_stop[1] is None:
            y_start_stop[1] = img.shape[0]
        # Compute the span of the region to be searched
        xspan = x_start_stop[1] - x_start_stop[0]
        yspan = y_start_stop[1] - y_start_stop[0]
        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
        # Compute the number of windows in x/y
        nx_windows = np.int(xspan / nx_pix_per_step) - 1
        ny_windows = np.int(yspan / ny_pix_per_step) - 1
        # Initialize a list to append window positions to
        window_list = []
        # Loop through finding x and y window positions
        # Note: you could vectorize this step, but in practice
        # you'll be considering windows one by one with your
        # classifier, so looping makes sense
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs * nx_pix_per_step + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = ys * ny_pix_per_step + y_start_stop[0]
                endy = starty + xy_window[1]

                # Append window position to list
                window_list.append(((startx, starty), (endx, endy)))
        # Return the list of windows
        return window_list

    # Define a function to draw bounding boxes
    def draw_boxes(self, img, bboxes, color=(0, 0, 255), thick=6):
        # Make a copy of the image
        imcopy = np.copy(img)
        # Iterate through the bounding boxes
        for bbox in bboxes:
            # IN HERE, WANT TO COMBINE BOXES
            # Draw a rectangle given bbox coordinates
            # x, y = zip(*bbox)
            # center = (max(x) + min(x)) / 2., (max(y) + min(y)) / 2.
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        # Return the image copy with boxes drawn
        return imcopy


def get_training_examples():
    cars = []
    notcars = []
    for root, dirs, files in os.walk('.'):
        for name in files:
            if name.endswith((".png", ".jpg")):
                if 'non-vehicles' in root:
                    notcars.append(os.path.join(root, name))
                elif 'vehicles' in root:
                    cars.append(os.path.join(root, name))
    return cars, notcars


def get_fit_model():
    cars, notcars = get_training_examples()

    ### TODO: Tweak these parameters and see how the results change.
    detect = DetectVehicles(config.color_space, config.spatial_size, config.hist_bins, config.orient, False,
                            config.pix_per_cell, config.cell_per_block, config.hog_channel,
                            config.spatial_feat, config.hist_feat, config.hog_feat)
    car_features = detect.extract_features(cars)
    notcar_features = detect.extract_features(notcars)
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:', config.orient, 'orientations', config.pix_per_cell,
          'pixels per cell and', config.cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    return svc, X_scaler


def vid_pipe(path='project_video.mp4', svc=None, X_scaler=None):
    from moviepy.editor import VideoFileClip
    clip = VideoFileClip(path)
    if svc is None:
        svc, X_scaler = get_fit_model()
    det = DetectVehicles(config.color_space, config.spatial_size, config.hist_bins, config.orient, False,
                         config.pix_per_cell, config.cell_per_block, config.hog_channel,
                         config.spatial_feat, config.hist_feat, config.hog_feat, fit_model=svc, X_scaler=X_scaler,
                         scale_img=False)
    output = clip.fl_image(det.draw_on_image)
    output.write_videofile('project_video_annotated.mp4', audio=False)
    return svc, X_scaler, det


def test_pipe(svc, X_scaler):
    import imageio
    det = DetectVehicles(config.color_space, config.spatial_size, config.hist_bins, config.orient, False,
                         config.pix_per_cell, config.cell_per_block, config.hog_channel,
                         config.spatial_feat, config.hist_feat, config.hog_feat, fit_model=svc, X_scaler=X_scaler)
    import glob
    for img in glob.glob('test_images/*'):
        print(img)
        image = imageio.imread(img)
        new = det.draw_on_image(image)
        cv2.imwrite('final_{0}.png'.format(img.split('/')[-1].split('.')[0]), new)


def centroid_function(detections, img):
    centroid_rectangles = []

    heat_map = np.zeros_like(img)[:, :, 0]

    for (x1, y1, x2, y2) in detections:
        heat_map[y1:y2, x1:x2] += 10

    heat_map.astype("uint8")

    _, binary = cv2.threshold(heat_map, 11, 255, cv2.THRESH_BINARY);

    _, contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        rectangle = cv2.boundingRect(contour)
        if rectangle[2] < 50 or rectangle[3] < 50: continue
        x, y, w, h = rectangle
        centroid_rectangles.append([x, y, x + w, y + h])

    return centroid_rectangles
