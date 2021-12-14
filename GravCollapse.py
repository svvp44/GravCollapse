import numpy as np 
import matplotlib.pyplot as plt
import scipy
from numpy import random 


n=3 #number of particles
E=0 #dampening factor
G=6.67408*10**(-11)  #gravitational constant

simtime=100*365*24*60*60

dt=5*24*3600
timesteps=40

pos=np.zeros((n,3)) #zero array for positions set up as [[x,y,z],[x,y,z],....]
mass=np.zeros((n,3)) #zero array for masses
vel=np.zeros((n,3)) #zero array for velocity
force=np.zeros((n,3)) #zero array for force

potential=np.array([0])

pos[0][0]=-743205035.8 #pos of the sun
mass[0]= (1.9889*10**30) #mass of the sun
vel[0][0]=-0.0911351706
vel[0][1]=-12.705823

mass[1]= (5.9742*10**24) #mass of the earth
#vel[1][1]=np.sqrt((G*mass[0][0])/pos[1][0]) #initial velocity of the earth
vel[1][0]=2544
vel[1][1]=29679
pos[1][0]=(1.4960*10**(11))-742757474 #distance from earth to sun so earth x=...

#Initial conditions for jupiter
mass[2]=(1.898*10**27)
pos[2][0]=((7.7833*10**11)-742757474)
vel[2][0]=93.2
vel[2][1]=12999.7


mass2=np.zeros((n))
mass2[0]= (1.9889*10**30)
mass2[1]= (5.9742*10**24)
mass2[2]=(1.898*10**27)
#FIND V 5 DAYS Prev

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

def potentials(m1, m2, x1, x2):
    U=-G*(m1*m2)/(flatdist(x1-x2))
    return U

def kineticEnergySingle(m,v):
    E=0.5*m*v**2
    return E

def kineticEnergyTotal(mass,currentv):
    totalE=0
    
    
    
    for i in range(len(mass)):
        flatv=flatdist(currentv[i])
        
        totalE += kineticEnergySingle(mass[i], flatv)
        
    return totalE
    


kineticEnergys=[]


currentv=vel
currentpos=pos

def totalForcePotential():
    force=np.zeros((n,3)) #zero array for force
    potential=np.array([0.0])
    for x in range(n):
        
        for i in range(x,n):
            
            if i!=x:
                
                
                xiforce=forces(mass[x][0],mass[i][0],currentpos[x],currentpos[i])
            
                force[x]+= xiforce
                force[i]+= xiforce*-1
                
                xipotential=potentials(mass[x][0],mass[i][0],currentpos[x],currentpos[i])
                
                potential+=xipotential
    
    return force, potential

times=[0]

sunxs=[currentpos[0][0]]
sunys=[currentpos[0][1]]

earthxs=[currentpos[1][0]]
earthys=[currentpos[1][1]]

jupxs=[currentpos[2][0]]
jupys=[currentpos[2][1]]

sunEarthDist=[1.4960*10**(11)]


def distBetween(pos1, pos2):
    seperation=pos1-pos2
    
    return flatdist(seperation)



potentialsArray=[]

#CHANGE TO WHILE TIMES[-1]<SIMULATIONTIME AND WORK OUT THE SIMULATION TIME GIVEN A NUMBER OF ORBITS
while times[-1]<simtime:
#for a in range(timesteps):
    
    forcePotential=totalForcePotential()
    vplushalf = currentv + ((forcePotential[0])/mass)*dt    
    newpos= currentpos + vplushalf*dt
    
    kineticEnergys.append(kineticEnergyTotal(mass2,currentv))
    potentialsArray.append(forcePotential[1])
    
    #plt.scatter(currentpos[1][0],currentpos[1][1],c="b")
    #plt.scatter(currentv[1][0],currentv[1][1])
    #plt.scatter(currentpos[0][0],currentpos[0][1],c="r")
        
    currentpos =newpos
    currentv =vplushalf
    
    for i in range(0,len(mass2)):
        if i==0:
            sunxs.append(currentpos[0][0])
            sunys.append(currentpos[0][1])
        elif i==1:
            earthxs.append(currentpos[1][0])
            earthys.append(currentpos[1][1])
        elif i==2:
            jupxs.append(currentpos[2][0])
            jupys.append(currentpos[2][1])
    
    
    

    
    sunEarthDist.append(distBetween(currentpos[0],currentpos[1]))
    times.append((times[-1])+dt)
    
kineticEnergys.append(kineticEnergyTotal(mass2,currentv))
potentialsArray.append(totalForcePotential()[1])
    
totalEnergy=np.array(potentialsArray)+np.array(kineticEnergys)


earthxsAU=np.array(earthxs)/(1.4960*10**(11))
earthysAU=np.array(earthys)/(1.4960*10**(11))
sunxsAU=np.array(sunxs)/(1.4960*10**(11))
sunysAU=np.array(sunys)/(1.4960*10**(11))
jupxsAU=np.array(jupxs)/(1.4960*10**(11))
jupysAU=np.array(jupys)/(1.4960*10**(11))
timeYears=np.array(times)/(3.154e+7)


#SHOWS THE ORBIT OF THE SUN AROUND THE EARTH
fig1=plt.figure(1)

ax=fig1.add_subplot(111)
#fig1(figsize=(8, 8))
ax.set_aspect('equal')
ax.plot(earthxsAU,earthysAU, label="Earth")
ax.plot(jupxsAU, jupysAU,c="orange", label="Jupiter")
ax.plot(sunxsAU,sunysAU,c="r", label="Sun")
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

"""
#SHOWS EARTHS Y POSITION OVER TIME
fig2=plt.figure(2)
ax2=fig2.add_subplot(111)
ax2.plot(timeYears,earthysAU)
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Earth's y position (AU)")
'''
ax2.spines['bottom'].set_color('white')
ax2.spines['top'].set_color('white') 
ax2.spines['right'].set_color('white')
ax2.spines['left'].set_color('white')
ax2.xaxis.label.set_color('white')
ax2.yaxis.label.set_color('white')
ax2.tick_params(axis='x', colors='white')
ax2.tick_params(axis='y', colors='white')
'''
#fig2.savefig('earthypos.png', transparent=True)

print("The time period of the Earth is", times[-1], "seconds which is", times[-1]*3.17098e-8, "years.")

plt.show()
"""
fig3=plt.figure(3)
ax3=fig3.add_subplot(111)

sunEarthDist=np.array(sunEarthDist)/(1.4960*10**(11))

ax3.plot(timeYears,sunEarthDist)
ax3.set_xlabel("Time (Yr)")
ax3.set_ylabel("Earth's distance to the sun (AU)")
ax3.set_ylim(0,2)
ax3.set_title("Earth-Sun Distance (AU)")
plt.show()

"""
"""
#kineticEnergys plot

fig4=plt.figure(4)
ax4=fig4.add_subplot(111)

relativeKinetic=kineticEnergys/kineticEnergys[0]

ax4.plot(timeYears,kineticEnergys)
ax4.set_xlabel("Time (Yr)")
ax4.set_ylabel("Total Kinetic Energy of the System (J)")
#ax4.set_ylim(0,3e+35)
ax4.set_title("Kinetic Energy")
plt.show()

#potentialsArray plot
fig5=plt.figure(5)
ax5=fig5.add_subplot(111)

relativePotentials=(potentialsArray/potentialsArray[0])*-1

ax5.plot(timeYears,potentialsArray)
ax5.set_xlabel("Time (Yr)")
ax5.set_ylabel("Total Potential Energy of the System (J)")
ax5.set_title("Potential Energy")
#ax5.set_ylim(-4e+35,-2.5e+35)
plt.show()


#Total Energy Over Time
fig6=plt.figure(6)
ax6=fig6.add_subplot(111)

relativeTotal=totalEnergy/totalEnergy[0]

ax6.plot(timeYears,totalEnergy)
ax6.set_xlabel("Time (Yr)")
ax6.set_ylabel("Total Energy of the System (J)")
ax6.set_title("Total Energy")
#ax6.set_ylim(-2.0e+35,0)
plt.show()

percentageChange=((totalEnergy[-1]-totalEnergy[0])/totalEnergy[0])*100
percentageChange=round(float(percentageChange),5)
print("The change in total energy is", float(percentageChange), "%")
print(totalEnergy[-1], totalEnergy[0])

#7 min talk on the 12th then 3 mins questions, we will find out more, 

#Sun Pos
fig7=plt.figure(7)
ax7=fig7.add_subplot(111)
ax7.set_aspect('equal')
ax7.scatter(sunxsAU[0],sunysAU[0])
ax7.plot(sunxsAU,sunysAU)
ax7.set_xlabel("Sun X (AU)")
ax7.set_ylabel("Sun Y (AU)")
ax7.set_title("Sun Pos")

plt.show()
# DIVIDE THE KE AND POT E BY THE FIRST DATA POINT



