import numpy as np
import os
import glob
import time
import fitness
import random
from scipy.stats import rankdata
import matplotlib.pyplot as plt

### Blocks for the genetic algorithm
class geneticACG:
    def __init__(self, Ngen, Npop, target, mutationRate=0.3, dummyFitness = False, elitism = True):
        self.targetlegSigma = target[0]
        self.targetOUSigma = target[1]
        self.targetOUtau = target[2] 

        self.Ngen = Ngen
        self.Npop = Npop

        self.population = None
        self.fitness = np.ndarray((Npop,Ngen+1))
        self.genealogicTree = np.ndarray((Npop,Ngen+1,4,4))

        self.bestElement = None
        self.bestFitness = 0
        self.bestIndex = None

        self.generation = 0

        self.fitnessMemory=[] #Memory preventing superfluous fitness evaluations, quite convenient when using elitism
        self.populationMemory=[]

        self.progress = 0
        

        self.startTime = time.time()
        self.initialisePop()
        self.genealogicTree[:,0,:,:]=self.generation
        for gen in range(self.Ngen):
            self.calculateFitness(dummyFitness)
            if elitism and gen!=0 :
                self.elitism()
            #print(self.fitness[:,gen])
            self.reproduction()
            if gen!=self.Ngen - 1:
                self.mutation(mutationRate) #It's best if the last reproduction batch doesn't mutate, in so far as exploration is no longer needed
            self.genealogicTree[:,0,:,:]=self.population
            self.generation+=1
            
        self.calculateFitness(dummyFitness)
        #print(self.fitness[:,-1])
        #print(self.population)
        np.save('fitness.npy', self.fitness)
        np.save('fitnessMemory.npy', self.fitnessMemory)
        np.save('populationMemory.npy', self.populationMemory)
        np.save('genealogicTree.npy', self.genealogicTree)
        
        self.performance()

        
        
        

    def elitism(self):
        maxFitness= self.fitness[0,self.generation]
        minIndex = 0
        minFitness= self.fitness[0,self.generation]
        minIndex = 0
        for i in range(len(self.fitness[:,self.generation])):
            if self.fitness[i,self.generation]<minFitness:
                minFitness= self.fitness[i,self.generation]
                minIndex = i
                #print(i)
            if self.fitness[i,self.generation]>maxFitness:
                self.bestFitness = self.fitness[i,self.generation]
                self.bestElement = self.population[i,:,:]
                self.bestIndex = i
                #print(i)
        
        self.fitness[minIndex,self.generation] = self.bestFitness
        self.population[minIndex,:,:] = self.bestElement


    def initialisePop(self, initialPool = None):
        if initialPool==None:
            self.population = np.ndarray((self.Npop, 4,4))
            for pop in range(self.Npop):
                for stage in range(3):
                    backl = 2*random.random()
                    frontl = 2*random.random()
                    legSigma = self.targetlegSigma * 1.2 * random.random()
                    OUSigma = self.targetOUSigma * 1.2 * random.random()
                    self.population[pop,stage,:] = [backl, frontl, legSigma, OUSigma]
                self.population[pop,3,:] = [1,1,self.targetlegSigma,self.targetOUSigma]
        else :
            self.population = initialPool
        print(self.population)


    def calculateFitness(self, dummy=False):
    
        if not dummy :
            for pop in range(self.Npop):
                storedFitness = False
                for i in range(len(self.fitnessMemory)):
                    similar = True
                    for j in range(len(self.population[pop,:,:].flatten())): 
                        if self.population[pop,:,:].flatten()[j]==self.populationMemory[i].flatten()[j]:
                            similar = False
                    if similar:
                        index = i
                        storedFitness = True
                        print('similar')
                        break
                if not storedFitness :

                    chromosome = np.stack((self.population[pop,:,0],self.population[pop,:,0],self.population[pop,:,1],self.population[pop,:,1],self.population[pop,:,2],self.population[pop,:,3], self.targetOUtau*np.ones(4)), axis = 1)                
                    #print(chromosome)
                    ppo = fitness.PPOtraining(chromosome, verbose = 0)
                    self.fitness[pop,self.generation] = ppo.evaluateFitness()
                    
                    self.populationMemory.append(self.population[pop,:,:])
                    self.fitnessMemory.append(self.fitness[pop,self.generation])
                else :
                    self.fitness[pop,self.generation] = self.fitnessMemory[index]
                self.progress+=1   
                print('Progress : '+str(self.progress)+' / '+str((self.Ngen+1)*self.Npop))
                print('Estimated remaining time : '+str((time.time()-self.startTime)/self.progress*((self.Ngen+1)*self.Npop-self.progress)) )    
        else :
            for pop in range(self.Npop):  
                storedFitness = False
                for i in range(len(self.fitnessMemory)):
                    similar = True
                    for j in range(len(self.population[pop,:,:].flatten())): 
                        if self.population[pop,:,:].flatten()[j]==self.populationMemory[i].flatten()[j]:
                            similar = False
                    if similar:
                        index = i
                        storedFitness = True
                        print('similar')
                        break
                if not storedFitness :
                    self.fitness[pop,self.generation] = np.sum(self.population[pop,:,:]) 
                    self.populationMemory.append(self.population[pop,:,:])
                    self.fitnessMemory.append(self.fitness[pop,self.generation])
                else :
                    self.fitness[pop,self.generation] = self.fitnessMemory[index]   

                self.progress+=1   
                print('Progress : '+str(self.progress)+' / '+str((self.Ngen+1)*self.Npop))
                print('Estimated remaining time : '+str(int((time.time()-self.startTime)/self.progress*((self.Ngen+1)*self.Npop-self.progress))) )      
          

    def mutation(self, rate, mutationType = 1):
        """Applies mutation
            Note that the last stage (aka the target stage) can't be modified
        Args:
            rate : average number of expected mutation PER INDIVIDUAL
        """
        if mutationType == 1:
            for i in range(self.Npop):
                if i == self.bestIndex :
                    pass #Elite element shall not mutate
                for j in range(3):
                    for k in range(4):
                        if random.random()<(rate/(3*4)):
                            if k<=1 :
                                self.population[i,j,k] += random.gauss(0,0.25)
                            elif k==3 or k==4:
                                self.population[i,j,k] = self.population[i,j,k] * random.gauss(1,0.25)                 
            #np.clip(self.population,0.,2.)
            self.population[self.population<0]=0
            self.population[self.population>2]=2

    def reproduction(self, selectionType = 'rank', crossOverType = 'onePoint', crossOverRate = 0.2, stageOnlyCrossOverRate = 0.4):
        #Selection
  
        if selectionType == 'roulette':
            probabilities = self.fitness[:,self.generation]/np.sum(self.fitness[:,self.generation])
            selectees = np.random.choice(range(self.Npop), self.Npop, p = probabilities)

        elif selectionType == 'rank':
            ranks = (rankdata(self.fitness[:,self.generation],method='min') ).astype(int)
            weights = ranks #np.ones(self.Npop)/ranks
            #print(ranks)
            #probabilities = ((np.max(ranks)+1)*np.ones(self.Npop)-ranks)/np.sum(ranks) #
            probabilities = weights/np.sum(weights)
            #print(probabilities)
            #print(np.sum(probabilities))
            selectees = np.random.choice(range(self.Npop), self.Npop, p = probabilities)

        else :
            raise ValueError("Only roulette ('roulette') and rank-based ('rank') selection are implemented")

        #Checking best element 
        ranks = (rankdata(self.fitness[:,self.generation], method='min') - 1).astype(int)
        

        #Should be moved to elitism method !!!! 
        #bestIndex = np.where(ranks==0)[0][0]
        #print('bestIndex')
        #print(bestIndex)
        
        
        #if self.fitness[bestIndex,self.generation]>self.bestFitness :
        #    self.bestElement = self.population[bestIndex,:,:]
        #    self.bestFitness = self.fitness[bestIndex,self.generation]
        #    print('best fitness :')
        #    print(self.bestFitness)
        #Cross overs

        if crossOverType == 'onePoint' :
            #applies to approx 40% of the population by default
            selectees=selectees.tolist()
            pairs= []
            while len(selectees) > int(self.Npop*(1-crossOverRate)) :
                index1 = selectees.pop(random.randint(0, len(selectees)-1))
                index2 = selectees.pop(random.randint(0, len(selectees)-1))
                #index1 = np.delete(selectees,random.randint(0, len(selectees)-1))
                #index2 = np.delete(selectees,random.randint(0, len(selectees)-1))

                pairs.append([index1, index2])
            newPop = self.population
            progressIndex = 0
            for pair in pairs:
                if random.random()<stageOnlyCrossOverRate:
                    stageSep = random.randint(1,2)
                    newPop[progressIndex,:stageSep,:] = self.population[pair[1],:stageSep,:]
                    newPop[progressIndex+1,:stageSep,:] = self.population[pair[0],:stageSep,:]
                    newPop[progressIndex,stageSep:,:] = self.population[pair[1],stageSep:,:]
                    newPop[progressIndex+1,stageSep:,:] = self.population[pair[0],stageSep:,:]
                else :
                    swapIndex = random.randint(0,3)
                    newPop[progressIndex,:,:] = self.population[pair[0],:,:]
                    newPop[progressIndex+1,:,:] = self.population[pair[1],:,:]
                    newPop[progressIndex,:,swapIndex] = self.population[pair[1],:,swapIndex]
                    newPop[progressIndex+1,:,swapIndex] = self.population[pair[0],:,swapIndex]

                progressIndex += 2
        
            #Non cross-over individuals :
            for individual in selectees:
                newPop[progressIndex,:,:] = self.population[individual,:,:]
                progressIndex += 1
        else :
            raise ValueError("Only one point ('onePoint') crossovers are implemented")
        self.population = newPop

    def performance(self):
        meanfit = np.mean(self.fitness, axis=0)
        minfit = self.fitness.min(axis=0)
        maxfit = self.fitness.max(axis=0)
        #print(self.fitness)
        fig=plt.plot(range(self.Ngen+1),meanfit,range(self.Ngen+1),minfit,range(self.Ngen+1),maxfit)
        plt.xlabel('Generations')
        plt.ylabel('Scores range')
        
        plt.savefig("synthesis.png")
        plt.show()


        



if __name__ == '__main__':
    ga = geneticACG(30, 10, [0.16,0.24,0.2], mutationRate = 0.2, dummyFitness = False, elitism=True)

        


