import open3d as o3d
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import pandas as pd
import numpy as np

class cubePoints():
    def __init__(self, cooridinate, color):
        self.coordinate = cooridinate
        self.color = color

def getCubePoints():
    points = []

    #front
    front = [[0,0,0], [1,0,0], [0,0,1], [1,0,1]]
    for i in range(9):
        for j in range(9):
            point_pos = [ (i+1)*0.1 , 0 , (j+1)*0.1]
            #print(point_pos)
            points.append(cubePoints(point_pos, (0, 0, 255)))

    #back
    for i in range(9):
        for j in range(9):
            point_pos = [ (i+1)*0.1 , 1 , (j+1)*0.1]
            #print(point_pos)
            points.append(cubePoints(point_pos, (0, 255, 0)))

    #top
    for i in range(9):
        for j in range(9):
            point_pos = [ (i+1)*0.1 , (j+1)*0.1, 1]
            #print(point_pos)
            points.append(cubePoints(point_pos, (255, 0, 0)))


    #bottom
    for i in range(9):
        for j in range(9):
            point_pos = [ (i+1)*0.1 , (j+1)*0.1, 0]
            #print(point_pos)
            points.append(cubePoints(point_pos, (0, 255, 255)))

    #left
    for i in range(9):
        for j in range(9):
            point_pos = [0, (i+1)*0.1 , (j+1)*0.1]
            #print(point_pos)
            points.append(cubePoints(point_pos, (255, 0, 255)))

    #right
    for i in range(9):
        for j in range(9):
            point_pos = [1, (i+1)*0.1 , (j+1)*0.1]
            #print(point_pos)
            points.append(cubePoints(point_pos, (255, 255, 0)))

    return points


def getImgName(images_df):
    imgName = images_df["IMAGE_ID"].to_list()
    validImgName = []

    for i in range(len(imgName)):
        idx = imgName[i]
        try:
            fname = ((images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values)[0]
            validImgName.append(fname)
        except:
            continue

    validImgName = sorted(validImgName, key = lambda name: int(name[name.find('g')+1:name.find('.')]))

    return validImgName

def getImg(validImgName):
    validImg = []

    for i in range(len(validImgName)):
        rimg = cv2.imread("data/frames/"+validImgName[i])
        validImg.append(rimg)

    return validImg

def showCube(img, rotation, position):
    cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]]) 
    cubePoints = getCubePoints()

    cubePoints.sort(key= lambda point : np.linalg.norm((point.coordinate-position)), reverse=True)
    for i, point in enumerate(cubePoints):
        rotation_Mat = R.from_rotvec(rotation).as_matrix()
        pixelPoints = (cameraMatrix @ (rotation_Mat @ (cubePoints[i].coordinate-position).T)).T
        pixelPoints = (pixelPoints/pixelPoints[2])
        img = cv2.circle(img, (int(pixelPoints[0]), int(pixelPoints[1])), radius=5, color=cubePoints[i].color, thickness=-1)

    return img

def showVideo():
    images_df = pd.read_pickle("data/images.pkl")

    try: 
        print('Fetching camera pose information')
        poses_df = pd.DataFrame(pd.read_pickle("cameraPose.pkl")) # {'R_pred': R_pred, 'T_pred': T_pred, 'R_gt': R_gt, 'T_gt': T_gt}
        print('Camera pose information exist')
        R_pred = []
        T_pred = []
        print("Fetching all poses' rotation and translation")
        for i in range(len(poses_df['R_pred'])):
            R_pred.append(poses_df['R_pred'][i])
            T_pred.append(poses_df['T_pred'][i])
        R_pred = np.array(R_pred)
        T_pred = np.array(T_pred)

        print('Fetching valid images')
        validImgName = getImgName(images_df)
        validImg = getImg(validImgName)
        shape = (int(validImg[0].shape[1]), int(validImg[0].shape[0]))
    except:
        print('Camera pose information not found')
        print('Please run 2d3dmathcing.py first')
        return None

    print("Fetching valid images' rotatinos and translations")
    R_list = []
    T_list = []
    for i in range(len(validImgName)):
        idx = ((images_df.loc[images_df["NAME"] == validImgName[i]])["IMAGE_ID"].values)[0]
        fname = validImgName[i]
        rimg = cv2.imread("data/frames/"+fname,cv2.IMREAD_GRAYSCALE)

        R_list.append(R_pred[idx-1])
        T_list.append(T_pred[idx-1])

    print('Addind cube to valid images')
    for i in range(len(validImg)):
        validImg[i] = showCube(validImg[i], R_list[i], T_list[i])

    print('Creating video')
    video = cv2.VideoWriter("Cube_at_NTU_Gate_undistort.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 15, shape)

    for i in range(len(validImg)):
        video.write(validImg[i])

    cv2.destroyAllWindows()
    video.release()

if __name__ == '__main__':
    showVideo()