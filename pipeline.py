

""" Rubikloud take home problem """
import luigi
import csv
import sklearn
import pandas as pd
import collections
import os
import numpy as np
import scipy as sp
from keras import utils
import io,math
from sklearn.cross_validation import train_test_split

from sklearn import svm
from sklearn.svm import SVC

import pickle

#clf_doorprob=SVC(probability=True,kernel='sigmoid')

#class ExtractorDataParser:
    #"""Loads a CSV data file and provides functions for working with it"""

    #def __init__(self):
        #self.clear()

    #def clear(self):
        ## Records is a multidimensional dictionary of: records[frame_number][face_number][roi][signal] = value
        #self.records = collections.defaultdict(lambda: collections.defaultdict(list))
        #self.first_frame_number = sys.maxsize
        #self.last_frame_number = 0
        #self.number_frames = 0
        #self.meta_data = []
        #self.missing_records = []
        

class CleanDataTask(luigi.Task):
    """ Cleans the input CSV file by removing any rows without valid geo-coordinates.

        Output file should contain just the rows that have geo-coordinates and
        non-(0.0, 0.0) files.
    """
    tweet_file = luigi.Parameter(default='airline_tweets.csv')
    output_file = luigi.Parameter(default='clean_data.csv')
    
    
    tweet_data_dict = collections.defaultdict(list)
    
    output_data = luigi.Parameter(default=tweet_data_dict)

    # TODO...
    # Load the file contents into memeory
    with open(os.getcwd() + '/' + 'airline_tweets.csv', 'rU', encoding="ISO-8859-1") as datafile, open(os.getcwd() + '/' + 'clean_data.csv', 'w', encoding="ISO-8859-1") as write_file:
        csvreader = csv.reader(datafile)
        csvwriter = csv.writer(write_file)

        counter=0
        num_cols=0
        for record in csvreader:              
            if counter==0:
                counter+=1
                num_cols=len(record)
                continue
            counter+=1

            if record[15][:] == '' or record[15][:] == '[0.0, 0.0]':
                pass
            else:
                csvwriter.writerow(record)

    datafile.close()
    write_file.close()
    

class TrainingDataTask(luigi.Task):
    """ Extracts features/outcome variable in preparation for training a model.

        Output file should have columns corresponding to the training data:
        - y = airline_sentiment (coded as 0=negative, 1=neutral, 2=positive)
        - X = a one-hot coded column for each city in "cities.csv"
    """
    tweet_file = luigi.Parameter()
    cities_file = luigi.Parameter(default='cities.csv')
    output_file = luigi.Parameter(default='features.csv')
    
    city_data_dict = collections.defaultdict(list)
    uniq_city_data_dict = collections.defaultdict(list)
        

    def requires(self):
        return self.CleanDataTask  

    # TODO...
    def euclid(coords_1,coords_2):
        
        dist=math.sqrt(math.pow(math.fabs(coords_2[0]-coords_1[0]),2) + math.pow(math.fabs(coords_2[1]-coords_1[1]),2))                
        return dist
    
    #io.open("to-filter.txt","r", encoding="utf-8") as f:
    with io.open(os.getcwd() + '/' + 'cities.csv', 'r', encoding="ISO-8859-1") as datafile:
        csvreader = csv.reader(datafile, delimiter=',')
        #csvreader.next(inf)
        counter=0
        for record in csvreader:              
            if counter==0:
                counter+=1
                continue
            counter+=1

            if record[1][:] == '' :
                pass
            else:
                city_data_dict[record[0][:]]=record[1:]
                uniq_city_data_dict[record[1][:]]=record[1:]
        
    sentiments=[]   
    with io.open(os.getcwd() + '/' + 'clean_data.csv', 'rU', encoding="ISO-8859-1") as file:
        
        csvreader = csv.reader(file)
        #csvreader.next(inf)
        
        closest_city=[]
        uniq_city_coords=np.zeros((len(uniq_city_data_dict),2)) 
        for stuff in csvreader:   
            if not stuff:
                continue
            emotion=str(stuff[5]).strip()
            if emotion.find("negative") != -1:
                sentiments.append(0)   
            elif emotion.find("neutral") != -1:
                sentiments.append(1)
            else:
                sentiments.append(2)
            
            y=sentiments    
                
            min_dist=float("inf")  
            runing_coords_=[]
            runing_coords_.append(float(stuff[15].split(',')[0][1:]))
            runing_coords_.append(float(stuff[15].split(',')[-1][:-1]))            
            
            all_coord_pairs=[]
            for keys in uniq_city_data_dict.keys():
                                                                
                all_coord_pairs.append(float(uniq_city_data_dict[keys][3]))
                all_coord_pairs.append(float(uniq_city_data_dict[keys][4]))
                
                distance=euclid(runing_coords_,all_coord_pairs)
                all_coord_pairs=[]
                if distance < min_dist:
                    min_dist=distance
                    min_city_key=keys
                    
            closest_city.append(min_city_key)      

    #X=np.zeros((len(city_data_dict)))    
    #citys=[]
    #for cities in city_data_dict.items():
        #citys.append(cities[1])  
    
    #common_labels=[]
    cpy_closest_citiesa=closest_city.copy()
    cpy_closest_citiesb=closest_city.copy()
    comonlbls=np.zeros(len(closest_city))
    
    # group same cities under common integer values-= toward classifiction -> one hot encoding right after loop:
    for indx,val in enumerate(cpy_closest_citiesa):  
        if isinstance(val, int):
            continue
        for indx2,val2 in enumerate(cpy_closest_citiesb):
             
            if val == val2:
                comonlbls[indx2]=indx    
                cpy_closest_citiesa[indx2]=indx
                #cpy_closest_cities2.pop(indx2)
                            
    X=utils.to_categorical(comonlbls)
    
    #X=np.transpose(X)
    
    indx_1=0
    with  open(os.getcwd() + '/' + 'features.csv', 'w', encoding="ISO-8859-1") as write_file:
        for rows in range(X.shape[0]):
        
            data_n_label=np.append(X[rows],[y[i]],axis=0)
            indx_1+=1

            #csvreader = csv.reader(datafile)
            csvwriter = csv.writer(write_file)
    
            csvwriter.writerow(data_n_label)
    
        write_file.close()    
    
        
    


class TrainModelTask(luigi.Task):
    """ Trains a classifier to predict negative, neutral, positive
        based only on the input city.

        Output file should be the pickle'd model.
    """
    tweet_file = luigi.Parameter()
    output_file = luigi.Parameter(default='model.pkl')

    # TODO...
    
    #split the datset :    
    
    
    with io.open(os.getcwd() + '/' + 'features.csv', 'rU', encoding="ISO-8859-1") as file:

        csvreader = csv.reader(file)
        #csvreader.next(inf)

        Y=np.zeros((855))
        features=np.zeros((855,852)) 
        i=0
        for rows in csvreader:   
            j=0
            for data in rows[:-2]:
                features[i][j]=float(data)
                j+=1
            if(j >= 854):
                break
            Y[j]=float(rows[-1])    
            i+=1
            if(i == 854):
                break            
            
    
    
    # fit SVM model:
    clf_door.fit(X, y)
    #clf_doorprob.fit(Xtrain, ytrain)    
    
 
    # open the file for writing
    fileObject = open(output_file,'wb') 
    
    # this writes the object a to the
    # file named 'testfile'
    pickle.dump(a,fileObject)   
    
    # here we close the fileObject
    fileObject.close()    
    
    


class ScoreTask(luigi.Task):
    """ Uses the scored model to compute the sentiment for each city.

        Output file should be a four column CSV with columns:
        - city name
        - negative probability
        - neutral probability
        - positive probability
    """
    tweet_file = luigi.Parameter()
    output_file = luigi.Parameter(default='scores.csv')

    # TODO...


if __name__ == "__main__":
    luigi.run()
