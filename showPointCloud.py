import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d

def get_lineSet(rotation, position, cameraMatrix, epoch, size):

    # get four corner points and camera 
    points = np.array([[0, 0, 0.1], [1080/2, 0, 1], [1080/2, 1920/2, 1], [0, 1920/2, 1]])
    points = np.linalg.pinv(cameraMatrix)@points.T
    rotation = R.from_rotvec(rotation).as_matrix()
    points =  position.reshape(3, 1) + np.linalg.pinv(rotation)@points
    points = points.T 

    # get center and concate all
    all_points = np.ones((points.shape[0]+1, points.shape[1]))
    all_points[:-1, :] = points

    all_points[-1, :] = position

    # set up line
    line_set = o3d.geometry.LineSet(
        points = o3d.utility.Vector3dVector(all_points),
        lines = o3d.utility.Vector2iVector([[0,1], [1,2], [2, 3], [0, 3], [0, 4], [1, 4], [2, 4], [3, 4]])
    )

    # generate color
    color = [0, 0, 0]
    colors = np.tile(color, (8,1))

    line_set.colors = o3d.utility.Vector3dVector(colors)

    return (line_set, position)

def drawTrajectory(camera1, camera2):

    # set up line
    line_set = o3d.geometry.LineSet(
        points = o3d.utility.Vector3dVector([camera1, camera2]),
        lines = o3d.utility.Vector2iVector([[0,1]])
    )

    # generate color
    color = [1, 0, 0]

    line_set.colors = o3d.utility.Vector3dVector([color])

    return line_set

def findNextPos(currentPose, allPose):
    next_pos_index = min(range(len(allPose)), key= lambda theIndex : np.linalg.norm(allPose[theIndex]-currentPose))
    distance = np.linalg.norm(allPose[next_pos_index]-currentPose)

    return next_pos_index, distance

def sortPoses(cameraPose):
    sortedPose = []
    # left_most_pos_index = min(camera_pos, key= lambda theArray : theArray[0])
    left_most_pos_index = max(range(len(cameraPose)), key= lambda theIndex : np.linalg.norm(cameraPose[theIndex]))
    sortedPose.append(cameraPose[left_most_pos_index])
    del cameraPose[left_most_pos_index] 

    all_run_time = len(cameraPose)-1

    for i in range(all_run_time):
        left_most_pos_index, distance = findNextPos(sortedPose[-1], cameraPose)
        if(distance > 0.8):
            continue
        sortedPose.append(cameraPose[left_most_pos_index])
        del cameraPose[left_most_pos_index]

    return sortedPose


def displayPointcloud(model, rigidTranslation):
    points, pointColor = model
    pcd= o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(pointColor.astype(float)/255)

    display = [pcd]


    rotations, positions = rigidTranslation
    cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])    

    # get camaera pose and add into display
    cameraPoses = []
    for i in range(len(rotations)):
        oneCamera, cameraPose = get_lineSet(rotations[i], positions[i], cameraMatrix, i, len(positions))
        display.append(oneCamera)
        cameraPoses.append(cameraPose)

    # sort camera pose from one side to the other side
    sorted_pos = sortPoses(cameraPoses)

    # draw line between two camera pose as trajectory
    for i in range(len(sorted_pos)-1):
        traject = drawTrajectory(sorted_pos[i], sorted_pos[i+1])
        display.append(traject)

    # draw 3D point cloud
    o3d.visualization.draw_geometries(display)