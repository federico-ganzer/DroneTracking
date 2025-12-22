import numpy as np

CONFIG = {
    #Camera intrinsics
    'focal_length': 800, 
    'principal_point': (640, 360),  
    'camera_resolution': (1280, 720),  
    
    #Camera baseline
    'baseline': 20.0,
    
    #Trajectory parameters
    
    'drone_center': np.array([0, 0, 80]),
    'drone_radius': 30.0,
    'drone_angular_velocity': 0.15,
    'num_frames': 500,
    'frame_rate': 30,
    
    #Perturbation parameters
    'rotation_noise_deg': 1.0,
    'translation_noise': 0.1,
    
    #Detection noise
    'detection_noise': 0.5,
    
    #Refinement parameters
    'refinement_enabled': True,
    'stop_refinement': 300,
    'ba_window_size': 10,
    'ba_interval' : 20
}