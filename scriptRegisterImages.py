import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm


########
#      #
# Main #
#      #
########

if __name__ == '__main__':
    # Load Images
    img1 = cv2.imread('Images/Im1.JPG', 0)
    img2 = cv2.imread('Images/Im2.JPG', 0)
    img1 = img1[1000::2, 0:3000:2]
    img2 = img2[1000::2, 0:3000:2]

    # Vizu 1
    plt.figure(figsize=(15, 12))
    plt.imshow(img2, cmap=cm.gray)
    plt.show()

    # SIFT descriptors
    sift = cv2.xfeatures2d.SIFT_create()
    # instead of sift = cv2.sift, need to install opencv-contrib-python with pip install -U opencv-contrib-python==3.4.2.17

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Vizu 2
    img = cv2.drawKeypoints(img1, kp1, np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.figure(figsize=(15, 12))
    plt.imshow(img)
    plt.show()

    # FLANN parameters => Be careful, it takes 5-10min for 3000x4000 images, 30" for 1200x1500
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    pts1 = []
    pts2 = []

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.25 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    print(len(good))

    # Homography and warp
    M, mask = cv2.findHomography(np.int32(pts1), np.int32(pts2), cv2.RANSAC, 5.0)
    img1warped = cv2.warpPerspective(img1, M, dsize=(img1.shape[1], img1.shape[0]))

    # Vizu 3
    img1 = np.array(img1, dtype=np.int32)
    img2 = np.array(img2, dtype=np.int32)
    img1warped = np.array(img1warped, dtype=np.int32)

    plt.figure(figsize=(15, 20))
    plt.subplot(211)
    plt.imshow(np.abs(-img1 + img2))
    plt.colorbar()
    plt.subplot(212)
    plt.imshow(np.absolute(-img1warped + img2))
    plt.colorbar()
    plt.show()