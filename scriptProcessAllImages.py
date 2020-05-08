import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
import histogram_matching as hm

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

def hisEqulColor(img):
    ycrcb=cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
    channels=cv2.split(ycrcb)
    print(len(channels))
    cv2.equalizeHist(channels[0],channels[0])
    cv2.merge(channels,ycrcb)
    cv2.cvtColor(ycrcb,cv2.COLOR_YCR_CB2BGR,img)
    return img

def histogram_matching_rgb(img2,img_ref):
    reference_histogram = hm.ExactHistogramMatcher.get_histogram(img_ref)
    new_img2 = hm.ExactHistogramMatcher.match_image_to_histogram(img2, reference_histogram)
    return new_img2

########
#      #
# Main #
#      #
########

if __name__ == '__main__':

    # Data
    view = 'Vue2'
    folder='/home/julienlefevre/ownCloud/Documents/Misc/PhotosCroissance/'+view +'/'

    import os

    for root, dirs, files in os.walk(folder):
        for filename in files:
            print(filename)

    files.sort()
    NbImages=len(files)

    # Histogram specification
    histSpec=True
    keyWord="NoHistSpec"
    if histSpec:
        keyWord="HistSpec"

    # Reference Image
    if view=='Vue2':
        idx_ref=2
        threshold=0.1
    else:
        idx_ref=3
        thresdhold=1


    step=4
    img_ref=cv2.imread(folder + files[idx_ref], 0)
    img_ref = img_ref[::step,::step]

    allImages=[]
    allHomographies=[]
    allNbGood=np.zeros((NbImages-1,))
    allRMSE=np.zeros((NbImages-1,2))

    cpt=0
    for i in range(NbImages):
        img2 = cv2.imread(folder + files[i], 0)
        img2 = img2[::step, ::step]
        try:
            (img2warped,M,nbGood)=register2Images(img2, img_ref, ratio_Lowe=0.75)
            allImages.append(img2warped)
            allHomographies.append(M)
            allNbGood[cpt]=nbGood
            allRMSE[cpt,0]=np.mean(np.abs(-np.int32(img_ref)+np.int32(img2warped)))
            allRMSE[cpt,1]=np.mean(np.abs(-np.int32(img_ref)+np.int32(img2)))
            print('Step '+str(cpt) + ', RMSE, after/before, '+ str(allRMSE[cpt,:]))
            cpt=cpt+1
        except:
            print('Registration failed')

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

    # Homographies close to identity + translation

    distanceIdentity=np.zeros((len(allHomographies,)))
    translation=np.zeros((len(allHomographies),2))
    for i in range(len(allHomographies)):
        distanceIdentity[i]=np.sum((allHomographies[i][0:2, 0:2] - np.eye(2, 2)) ** 2)
        translation[i,:]=allHomographies[i][0:2,2].transpose()

    # Movie after registration
    ims = []
    for i in range(len(allRMSE)):
        if distanceIdentity[i]<threshold:
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


    # Movie before registration
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
    #folderToSave='/home/julienlefevre/ownCloud/Documents/Misc/PhotosCroissance/Videos/'



    folderToSave='Videos/'
    try:
        os.mkdir(folderToSave)
    except:
        print('folder ' + folderToSave + ' already exists')
    cpt=1

    os.system('rm ' +folderToSave + '*.jpg')
    os.system('rm ' +folderToSave + 'RegistrationOn'+view+keyWord+'.mp4')
    os.system('rm ' +folderToSave + 'RegistrationOff' + view + '.mp4')
    img_ref = cv2.imread(folder + files[idx_ref])
    img_ref = img_ref[::step, ::step]

    nzeros=np.floor(np.log10(NbImages))+1
    for i in range(len(allRMSE)):
        if distanceIdentity[i]<threshold:
            img2 = cv2.imread(folder + files[i])
            img2 = img2[::step, ::step]
            if histSpec:
                img2 = histogram_matching_rgb(img2, img_ref)
            tmpImage=cv2.warpPerspective(img2, allHomographies[i], dsize=(img2.shape[1], img2.shape[0]))
            cv2.imwrite(folderToSave+'photo'+str(cpt).zfill(int(nzeros)) +'.jpg', tmpImage)
            cpt=cpt+1

    os.system('ffmpeg -f image2 -framerate 4 -i '+folderToSave+ 'photo%2d.jpg -r 5 '+folderToSave+'RegistrationOn'+view +keyWord+'.mp4')


    # Movie without registration

    #folderToSave = '/home/julienlefevre/ownCloud/Documents/Misc/PhotosCroissance/Videos/'
    cpt = 1
    nzeros = np.floor(np.log10(NbImages)) + 1
    for i in range(len(allRMSE)):
        if distanceIdentity[i] < threshold:
            img2 = cv2.imread(folder + files[i])
            img2 = img2[::step, ::step]
            cv2.imwrite(folderToSave + 'photo' + str(cpt).zfill(int(nzeros)) + '.jpg', img2)
            cpt = cpt + 1

    os.system('ffmpeg -f image2 -framerate 4 -i ' +folderToSave+ 'photo%2d.jpg -r 5 '+folderToSave+ 'RegistrationOff'+view+ '.mp4')

    # # Egalisation histogramme
    # img2 = cv2.imread(folder + files[0])
    # img2 = cv2.imread(folder + files[18])
    # plt.imshow(cv2.cvtColor(hisEqulColor(img2), cv2.COLOR_BGR2RGB))