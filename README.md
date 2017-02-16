
**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./test_images/final_405.png
[image2]: ./test_images/final_test5.png
[image3]: ./test_images/final_test5.png
[image4]: ./non-vehicles/GTI/image1043.png
[image5]: ./vehicles/GTI_Far/image0074.png
[image6]: ./heatmap.png
[image7]: ./hog_image.png
[image8]: ./video_output_images/final_0.png
[video1]: ./project_video_annotated.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines #118 through #141 of the file called `main.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:


![alt text][image4]
![alt text][image5]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=7`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

Here are the hog features:
![alt text][image7]

![alt text][image2]


####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of the different parameters on 6 test images, along with testing different sliding windows and heat thresholds. Also, when running my train_test_split I tracked the better performing options. All of these things added up to my eventual solution.

During my test, I had >99% accuracy
> ('Using:', 7, 'orientations', 8, 'pixels per cell and', 2, 'cells per block')
> ('Feature vector length:', 7284)
> (25.98, 'Seconds to train SVC...')
> ('Test Accuracy of SVC = ', 0.9941)

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained LinearSVC() using scikit-learn. This is done in the `get_fit_model()` function in `main.py`

I used HOG, along with spatial and histogram features.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

One of the biggest things I learned was to do two separate sliding scales. This allowed me to find better overlap on real cars and get a better view of those cars, then I could properly use a heatmap to get rid of false positives.

`windows = self.slide_window(draw_image, x_start_stop=[690, None], y_start_stop=[375, 430],
                                    xy_window=(110, 90), xy_overlap=config.xy_overlap)`

`windows += self.slide_window(draw_image, x_start_stop=[760, None], y_start_stop=[375, 560],
                                         xy_window=(110, 90), xy_overlap=config.xy_overlap)`
                                         
In this code in `main.py` I implemented the slide_window function twice on different pieces of the image.

Here's an example of all the boxes it creates:
![alt text][image8]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. I did a heatmap to find the boxes that laid on top of each other and only kept those that have >7 boxes found in an area.

In the functions `add_heat()` and `apply_threshold()` in `main.py` you can see that I've simply started +1 brightness to every pixel in all the boxes that were found. Then, knowing there should be significant overlap, I only use the "hottest" areas

Here are some example images:


Here's my heatmap
![alt text][image6]

![alt text][image2]

![alt text][image3]

-------------------

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://youtu.be/Eo9yJOlh2mM)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. I did multiple sliding windows so for my heatmap I specifically looked for areas that had multiple boxes overlappnig. I played around with my pipeline and landed on 5-6 boxes to get to a good option.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

I did all of this in the `add_heat()` and `apply_threshold()` functions in `main.py`


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The project was relatively straightforward using the material from the lectures. I would like to include more frame-by-frame detections, and I would like to split up the image more and also create search_windows that are doing smaller windows for further out in the frame and larger windows for vehicles close by. This will fail in low lighting and a lot of other situations. It has too many false positives still and would brake the car too often. I'd like to use more deep learning and more images to detect the cars better.