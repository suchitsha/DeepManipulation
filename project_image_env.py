# {'exits': [],
#  'item_type': 'script_item',
#  'parameters': [],
#  'position': (200, 20, -1, -1),
#  'transitions': []}
### end of header
import pickle
import pyutils.matrix as pm
import cv2
import numpy as np
from odb_interface import odb_utils

def init(self):
     pass

def execute(self):
    object_of_interest = "tray"

    fd = open("/home/shar_sc/exec_env/suchit/metainfo_20170315_133418.txt")
    metainfo = pickle.load(fd)
    fd.close()
      
    fd = open("/home/shar_sc/exec_env/suchit/q_act_20170315_133930.txt")
    q = map(float, fd.read().strip().split("\n"))
    fd.close()
    cfg = app.rave.get_config_dict()
    cfg["torso"] = q[0:3]
    cfg["right_arm"] = q[3:10]
    cfg["left_arm"] = q[10:17]
    cfg["head"] = q[17:19]
    app.rave.set_config_dict(cfg)
    app.rave.set_frame("Justin", metainfo["Justin_frame"])
    head = app.rave.get_manip_frame("head")
    roi_frame = odb_utils.float16_to_array(app.wsr.object_store["tray"]["toolframe"]) 
    object_frame = app.rave.get_frame(object_of_interest)
    target_frame = dot(object_frame, roi_frame)
    app.rave.add_coord("tray", object_frame)
    center = dot(head, metainfo["ext"])
    app.center = center
    app.rave.add_coord("center", center)

    img_name = app.out_dir + 'heat_map_th4.jpeg' 
    print "Reading Image" , img_name
    image = cv2.imread(img_name)
    i_height,i_width = image.shape[:2]
    fov_y, fov_x = metainfo["fov"]
    ratio_width = (fov_y)/i_width
    ratio_height = (fov_x)/i_height
    app.rave.clear_graphic_handles()
    
    n_particles = 2
    #distribution = "uniform"
    #rand = getattr(random, distribution)

    particles_x = []
    particles_y = []
    #resolution of the boxes to be drawn
    resolution_w = 60#30
    resolution_h = 60#30    
    for i in xrange(-i_width/2, i_width/2, resolution_w):
        for j in xrange(-i_height/2, i_height/2, resolution_h):
            sleep(0.00001)
            value = image[j+i_height/2][i+i_width/2]
                
            #position on the board         
            origin = dot(center, pm.roty(ratio_height*i*pi/180))       
            origin = dot(origin, pm.rotx(-ratio_width*j*pi/180))
            destination = dot(origin, pm.txyz(0,0,10))
            #app.rave.add_coord("origin", origin)
            #app.rave.add_coord("destination", destination)

            #draw box of size proportional to probability
            hits = app.rave.check_ray_collision(origin, destination, [object_of_interest])
            if object_of_interest not in hits.keys():
                continue
            position = hits[object_of_interest][0:3]
            direction = hits[object_of_interest][3:]
            #app.rave.add_coord("tray_hit", pm.txyz(*position))
            b_name = "box_%i_%i" %(i, j)
            app.rave.draw_box(b_name, pm.txyz(*position), [0.01]*3, 2, value[::-1]/255.)
            
            #n_particles = (value[0]/255.)*n_particles_ratio        
            #particles = (rand(position[0], position[0] + 0.01 , size=n_particles), rand(position[1], position[0] + 0.01, size=n_particles)
            #chance = np.random.lognormal(1,0.5)
            #chance = min((1., chance))
            #print chance
            
            #filter only dirty area
            #without this threshhold areas with low probability will also have few points
            thresh_green = 150
            #draw number of particles prpotional to intensity of color
            n = n_particles*(value[1]/255.)
            #so that multiple particles ddont overlap
            displace_x = 0.005
            displace_y = 0.005            
            #print value
            if value[1] > thresh_green:
                position_on_tray = dot(inv(target_frame), pm.txyz(*position))
                for x in range(int(n)):                    
                    particles_x.append(position_on_tray[0,3] + x*displace_x)
                    particles_y.append(position_on_tray[1,3] + x*displace_y)
            #distance = norm(dot(inv(origin), pm.txyz(*position))[0:3,3])
            #print distance
    particles = np.vstack((particles_x, particles_y))
    app.particles = particles            
    print app.particles
        