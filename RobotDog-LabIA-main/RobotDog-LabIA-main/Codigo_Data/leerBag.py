import rosbag
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d

bag = rosbag.Bag("/home/labia-001/recordings_from_go1/rslidar_only_20250917_005504_1.bag")
for topic, msg, t in bag.read_messages(topics=["/rslidar_points"]):
    pc = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
    points = list(pc)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([cloud])
    break
bag.close()
