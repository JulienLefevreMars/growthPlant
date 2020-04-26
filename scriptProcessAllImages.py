import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation

def register2Images(img1,img2,ratio_Lowe=0.75):
    # SIFT descriptors
    sift = cv2.xfeatures2d.SIFT_create()
    # instead of sift = cv2.sift, need to install opencv-contrib-python with pip install -U opencv-contrib-python==3.4.2.17

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters => Be careful, it takes 5-10min for 3000x4000 images, 30" for 1200x1500
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # ratio =[]
    # tmpPts1 = []
    # tmpPts2 = []
    #
    # # ratio test as per Lowe's paper
    # for i, (m, n) in enumerate(matches):
    #     ratio.append(m.distance / n.distance)
    #     tmpPts2.append(kp2[m.trainIdx].pt)
    #     tmpPts1.append(kp1[m.queryIdx].pt)
    # idx = np.argsort(np.array(ratio))
    # pts1=[]
    # pts2=[]
    # for i in idx[0:nbGood]:
    #     pts1.append(tmpPts1[i])
    #     pts2.append(tmpPts2[i])
    # print('Min/max good points: ' + str(ratio[idx[0]]) + ', ' + str(ratio[idx[nbGood]])+ '\n')

    good = []
    pts1 = []
    pts2 = []

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < ratio_Lowe * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    print(len(good))

    # Homography and warp
    M, mask = cv2.findHomography(np.int32(pts1), np.int32(pts2), cv2.RANSAC, 5.0)
    img1warped = cv2.warpPerspective(img1, M, dsize=(img1.shape[1], img1.shape[0]))

    return (img1warped,M,len(good))


########
#      #
# Main #
#      #
########

if __name__ == '__main__':

    # Data
    folder='/home/julienlefevre/ownCloud/Documents/Misc/PhotosCroissance/Vue2/'

    import os

    folder = '/home/julienlefevre/ownCloud/Documents/Misc/PhotosCroissance/Vue2/'

    for root, dirs, files in os.walk(folder):
        for filename in files:
            print(filename)

    files.sort()
    NbImages=len(files)


    # Reference Image
    # Choice1
    #img_ref = cv2.imread(folder + files[0], 0)
    #img_ref = img_ref[1000::2, 0:3000:2]
    idx_ref=2
    step=4
    img_ref=cv2.imread(folder + files[idx_ref], 0)
    img_ref = img_ref[::step,::step]

    allImages=[]
    allHomographies=[]
    allNbGood=np.zeros((NbImages-1,))
    allRMSE=np.zeros((NbImages-1,2))

    #for i in range(1):
    for i in range(NbImages):
        img2 = cv2.imread(folder + files[i], 0)
        img2 = img2[::step, ::step]

        (img2warped,M,nbGood)=register2Images(img2, img_ref, ratio_Lowe=0.75)
        allImages.append(img2warped)
        allHomographies.append(M)
        allNbGood[i]=nbGood
        allRMSE[i,0]=np.mean(np.abs(-np.int32(img_ref)+np.int32(img2warped)))
        allRMSE[i,1]=np.mean(np.abs(-np.int32(img_ref)+np.int32(img2)))
        print('Etape '+str(i) + ', RMSE, after/before, '+ str(allRMSE[i,:]))

    # meanImage=np.zeros(img_ref.shape,dtype=np.int32)
    #
    # for i in range(len(allImages)):
    #     meanImage=meanImage+allImages[i]
    #
    # plt.imshow(meanImage)
    #
    # for i in range(len(allImages)):
    #     plt.imshow(allImages[i])

    # On garde les images où RMSE[i,0]<RMSE[i,1] => pas un bon critère

    # Homographies proche de identité + translation

    distanceIdentity=np.zeros((len(allHomographies,)))
    translation=np.zeros((len(allHomographies),2))
    for i in range(len(allHomographies)):
        distanceIdentity[i]=np.sum((allHomographies[i][0:2, 0:2] - np.eye(2, 2)) ** 2)
        translation[i,:]=allHomographies[i][0:2,2].transpose()

    # Film après recalage
    ims = []
    for i in range(len(allRMSE)):
        if distanceIdentity[i]<0.1:
            im = plt.imshow(allImages[i], 'gray',animated=True)
        # if (allRMSE[i,0]<allRMSE[i,1]):
        #     im.figure.set_facecolor('red')
        #     #im.figure.title('Avant: '+str(allRMSE[i,0]) + ' Après:' + str(allRMSE[i,1]) )
        # else:
        #     im.figure.set_facecolor('green')
        ims.append([im])

    fig=plt.figure(figsize=(15,12))
    ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True,
                                    repeat_delay=1000)

    #plt.rcParams['animation.ffmpeg_path'] = '/home/julienlefevre/Documents/Softs/anaconda3/bin/ffmpeg'
    #ani.save('demo1.avi',writer=writer)
    #ani.save('demo1.gif', writer='imagemagick')

    plt.show()


    # Film avant recalage
    ims =[]
    for i in range(NbImages):
        img2 = cv2.imread(folder + files[i], 0)
        img2 = img2[::step, ::step]
        im = plt.imshow(img2, 'gray', animated=True)
        ims.append([im])

    fig=plt.figure(figsize=(15,12))
    ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True,
                                    repeat_delay=1000)
    #ani.save('demo2.gif', writer='imagemagick')
    plt.show()

    # # Sauver les images pour faire un film
    # folderToSave='/home/julienlefevre/ownCloud/Documents/Misc/PhotosCroissance/Videos/'
    # cpt=1
    # nzeros=np.floor(np.log10(NbImages))+1
    # for i in range(len(allRMSE)):
    #     if distanceIdentity[i]<0.1:
    #         cv2.imwrite(folderToSave+'photo'+str(cpt).zfill(int(nzeros)) +'.jpg', allImages[i])
    #         cpt=cpt+1

    # Sauver les images pour faire un film: en couleur
    folderToSave='/home/julienlefevre/ownCloud/Documents/Misc/PhotosCroissance/Videos/'
    cpt=1
    nzeros=np.floor(np.log10(NbImages))+1
    for i in range(len(allRMSE)):
        if distanceIdentity[i]<0.1:
            img2 = cv2.imread(folder + files[i])
            img2 = img2[::step, ::step]
            tmpImage=cv2.warpPerspective(img2, allHomographies[i], dsize=(img2.shape[1], img2.shape[0]))
            cv2.imwrite(folderToSave+'photo'+str(cpt).zfill(int(nzeros)) +'.jpg', tmpImage)
            cpt=cpt+1

    # Film sans recalage

    folderToSave = '/home/julienlefevre/ownCloud/Documents/Misc/PhotosCroissance/Videos/'
    cpt = 1
    nzeros = np.floor(np.log10(NbImages)) + 1
    for i in range(len(allRMSE)):
        if distanceIdentity[i] < 0.1:
            img2 = cv2.imread(folder + files[i])
            img2 = img2[::step, ::step]
            cv2.imwrite(folderToSave + 'photo' + str(cpt).zfill(int(nzeros)) + '.jpg', img2)
            cpt = cpt + 1