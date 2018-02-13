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
import io


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
        #csvreader.next(inf)
        counter=0
        num_cols=0
        for record in csvreader:              
            if counter==0:
                counter+=1
                num_cols=len(record)
                continue
            counter+=1
            #if record[0][0] == '#':                                   
                #self.process_comment_record(record)
                #continue    
            #print(record[15][:])    
           
            if record[15][:] == '' or record[15][:] == '[0.0, 0.0]':
                pass
            else:
                
                #record[entry]=record[entry].strip(',')
                #tweet_data_dict[record[0][:]]=record
                csvwriter.writerow(record)
                
        wait=-1                      
    bolix=0  
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
        

    def requires(self):
        return self.CleanDataTask  

    # TODO...
    
    def euclid(coords_1,coords_2):
        
        dist=math.sqrt((math.abs(coords_2[1]-coords_1[1]))^2 + (math.abs(coords_2[1]-coords_1[1]))^2)                
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
            #if record[0][0] == '#':                                   
                    #self.process_comment_record(record)
                    #continue    
            #print(record[1][:])    

            if record[1][:] == '' :
                pass
            else:
                city_data_dict[record[0][:]]=record[1:]

        wait=-1  
        
    sentiments=[]   
    with io.open(os.getcwd() + '/' + 'clean_data.csv', 'rU', encoding="ISO-8859-1") as file:
        
        csvreader = csv.reader(file)
        #csvreader.next(inf)
        
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
                
            min_dist=float("inf")  
            #coords_b.append()
            #stuff=-1
            #for cities in city_data_dict.items():
                #coors_a=cities[15]
                #if ( euclid() )

    X=np.zeros((len(city_data_dict)))    
    citys=[]
    for cities in city_data_dict.items():

        citys.append(cities[1])
        
    X=utils.to_categorical(citys[1])
        
            
    bla=0
    
    



class TrainModelTask(luigi.Task):
    """ Trains a classifier to predict negative, neutral, positive
        based only on the input city.

        Output file should be the pickle'd model.
    """
    tweet_file = luigi.Parameter()
    output_file = luigi.Parameter(default='model.pkl')

    # TODO...


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
