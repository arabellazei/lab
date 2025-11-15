No other software outside of what was given with the initial lab is required.
The main functions in each file have been edited slightly (parameters / return types changed) and they should run as is - values in the config file can be changed. 

1. inverse_geometry.py
   We implemented computeqgrasppose in inverse_geometry using the inverse kinematics algorithm from the tutorial. We do both tasks at the same time by projecting into the null space of the first task. 

2. path.py
   In path.py we implemented path planning using the rrt algroithm on cube locations. We generate a sample cube space (within certain bounds such as near start and end goal and within robot reach), and find the nearest cube in the graph to interpolate. To ensure grasping config, we discretise the path and use computeqgrasppose to check the validity of the path, as well as collision checking.

3. control.py
   In control.py, we managed to achieve motion of the robot, but were unable to achieve the cube grasping functionality. The trajectory is planned using cubic Hermite interpolation.