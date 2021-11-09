import numpy as np 
import matplotlib.pyplot as plt
import scipy
from numpy import random 


n=2 #number of particles
E=1 #dampening factor
G=6.67408*10**(-11)  #gravitational constant


dt=10*24*3600
timesteps=40

pos=np.zeros((n,3)) #zero array for positions set up as [[x,y,z],[x,y,z],....]
mass=np.zeros((n,3)) #zero array for masses
vel=np.zeros((n,3)) #zero array for velocity
force=np.zeros((n,3)) #zero array for force



pos[1][0]=1.4960*10**(11) #distance from earth to sun so earth x=...
mass[0]= 1.9889*10**30 #mass of the sun
mass[1]= 5.9742*10**24 #mass of the earth
vel[0]=0.0 #initial velocity of the sun
#vel[1][1]=np.sqrt((G*mass[0][0])/pos[1][0]) #initial velocity of the earth
vel[1][0]=2544
vel[1][1]=29679



#FIND V 5 DAYS PRev

#def positions(n): 

#    for i in range(n): 

#        for i in range(2): 

#            pos[n,i]=random.randint(100) #make floats from –1 and 1 or 0 to 1 

#        return pos 

#use a seed so if code is re ran it’s the same starting positions 

  
#def masses(n): 

#    for i in range(n): 

#        mass[i]=random.randint(100) 

#    return mass 

# don’t do random masses, start all the same mass 


#def vels(n): 

#    for i in range(n): 

#        vel[i]=random.randint(100) 

#    return vel 

# have conditioning for velocities, do this later 


def flatdist(array):
    xdist=array[0]
    ydist=array[1]
    zdist=array[2]
        
    dist=np.sqrt(xdist**2 + ydist**2 +zdist**2)
        
    return np.array([dist])


def forces(m1, m2, x1, x2): #finds the force between 2 particles

    temp=-1*((m1*m2)*(x1-x2))/((((flatdist((x1-x2)))**2)+E**2)**1.5) 
    F=G*temp
    
    return F


currentv=vel
currentpos=pos

def totalforce():
    force=np.zeros((n,3)) #zero array for force
    for x in range(n):
        
        for i in range(x,n):
            
            if i!=x:
                
                xiforce=forces(mass[x][0],mass[i][0],currentpos[x],currentpos[i])
                #print(xiforce)
                
                force[x]+= xiforce
                force[i]+= xiforce*-1
    
    return force

times=[0]

earthxs=[currentpos[1][0]]
earthys=[currentpos[1][1]]

sunxs=[currentpos[0][0]]
sunys=[currentpos[0][1]]

simtime=365*24*60*60

#CHANGE TO WHILE TIMES[-1]<SIMULATIONTIME AND WORK OUT THE SIMULATION TIME GIVEN A NUMBER OF ORBITS
while times[-1]<simtime:
#for a in range(timesteps):

    vplushalf = currentv + (totalforce()/mass)*dt    
    newpos= currentpos + vplushalf*dt
    
    
    
    
    #plt.scatter(currentpos[1][0],currentpos[1][1],c="b")
    #plt.scatter(currentv[1][0],currentv[1][1])
    #plt.scatter(currentpos[0][0],currentpos[0][1],c="r")
        
    currentpos =newpos
    currentv =vplushalf
    
    earthxs.append(currentpos[1][0])
    earthys.append(currentpos[1][1])
    
    sunxs.append(currentpos[0][0])
    sunys.append(currentpos[0][1])
    times.append((times[-1])+dt)


earthxsAU=np.array(earthxs)/(1.4960*10**(11))
earthysAU=np.array(earthys)/(1.4960*10**(11))
sunxsAU=np.array(sunxs)/(1.4960*10**(11))
sunysAU=np.array(sunys)/(1.4960*10**(11))

#SHOWS THE ORBIT OF THE SUN AROUND THE EARTH
fig1=plt.figure(1)
ax=fig1.add_subplot(111)

ax.set_aspect('equal')
ax.plot(earthxsAU,earthysAU, label="Earth")
ax.scatter(sunxsAU,sunysAU,c="r", label="Sun")
ax.legend()
ax.set_xlabel("X Position (AU)")
ax.set_ylabel("Y Position (AU)")
"""
ax.spines['bottom'].set_color('white')
ax.spines['top'].set_color('white') 
ax.spines['right'].set_color('white')
ax.spines['left'].set_color('white')
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
"""
#fig1.savefig('earthsun.png', transparent=True)
plt.show()


#SHOWS EARTHS Y POSITION OVER TIME
fig2=plt.figure(2)
ax2=fig2.add_subplot(111)
ax2.plot(times,earthysAU)
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Earth's y position (AU)")
"""
ax2.spines['bottom'].set_color('white')
ax2.spines['top'].set_color('white') 
ax2.spines['right'].set_color('white')
ax2.spines['left'].set_color('white')
ax2.xaxis.label.set_color('white')
ax2.yaxis.label.set_color('white')
ax2.tick_params(axis='x', colors='white')
ax2.tick_params(axis='y', colors='white')
"""
#fig2.savefig('earthypos.png', transparent=True)

print("The time period of the Earth is", times[-1], "seconds which is", times[-1]*3.17098e-8, "years.")

plt.show()



