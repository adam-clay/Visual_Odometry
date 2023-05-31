### Monocular Visual Odometry

    The challenge is to track the path of the camera using information from the videos captured. The algorithm is described here https://en.wikipedia.org/wiki/Visual_odometry


- A skeleton code is provided in `Odometry.py` , with necessaary structure for submitting to gradescope
- A sample video is given to test your methods and an evaluation code to benchmark your approaches
- The video is processed for lens distortion and converted to frames. The frames are numbered in the order they appear in the video
- Your algorithm will be evaluated on Gradescope autograder using another video which is not provided here
- The camera parameters for both the videos are present in the file `calib.txt`
- The ground truth for the the sample video is present in `gt_sequence.txt`. The process of evalutation and computing the final error from predictions is outlined in the evaluation notebook.

### Submisison Guidelines

The code submitted is evaluated on the second video and the resulting error is used to rank the submissions. The requirements for submission is 

1) .py file name Odometry.
2) Contain a class OdometryClass: 


    a. has a init function with video frames folder as input. 

    b. has a function called run to return predicted path.


Here is a sample code that the autograder uses to run the submissions

```{python}
    from Odometry import OdemotryClass 

    test_odometry = OdemotryClass(frame_path)
    path = test_odometry.run()
```

The following packages can be used in the challenge

    - Open CV
    - Scipy
    - Numpy
